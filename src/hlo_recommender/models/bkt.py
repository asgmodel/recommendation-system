"""Bayesian knowledge tracing with likelihood fitting and ordered assessment updates."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import numpy as np
from scipy.optimize import minimize
from ..data import validate_events
from ..evaluation import binary_metrics
from ..io import object_digest

@dataclass(frozen=True)
class BKTParameters:
    initial: float = 0.2
    learn: float = 0.1
    guess: float = 0.2
    slip: float = 0.1

    def __post_init__(self):
        if any((not 0 < x < 1 for x in asdict(self).values())) or self.guess + self.slip >= 1:
            raise ValueError('Invalid BKT probabilities')

    def predict(self, mastery: float) -> float:
        if not 0 <= mastery <= 1:
            raise ValueError('Invalid mastery probability')
        return mastery * (1 - self.slip) + (1 - mastery) * self.guess

    def update(self, mastery: float, correct: bool) -> float:
        if type(correct) is not bool:
            raise ValueError('Assessment outcome must be boolean')
        prediction = self.predict(mastery)
        posterior = mastery * (1 - self.slip) / prediction if correct else mastery * self.slip / (1 - prediction)
        return posterior + (1 - posterior) * self.learn

def ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (datetime.fromisoformat(r['timestamp']), r['event_id']))

def predict_events(rows: list[dict], params: BKTParameters) -> tuple[list[dict], dict]:
    validate_events(rows)
    states, predictions = ({}, [])
    for row in ordered(rows):
        key = (row['learner_id'], row['skill_id'])
        mastery = states.get(key, params.initial)
        probability = params.predict(mastery)
        states[key] = params.update(mastery, row['correct'])
        predictions.append({**row, 'probability': probability, 'mastery_before': mastery, 'mastery_after': states[key]})
    metrics = binary_metrics([r['correct'] for r in predictions], [r['probability'] for r in predictions])
    return (predictions, metrics)

def fit_bkt(rows: list[dict], *, seed=42, restarts=5) -> tuple[BKTParameters, dict]:
    validate_events(rows)
    if type(restarts) is not int or restarts < 1:
        raise ValueError('BKT fitting requires at least one restart')
    if len(rows) < 20 or len({r['learner_id'] for r in rows}) < 2:
        raise ValueError('At least 20 assessments from two learners are required')
    sequences = {}
    for row in ordered(rows):
        sequences.setdefault((row['learner_id'], row['skill_id']), []).append(row['correct'])

    def objective(values):
        params = BKTParameters(*map(float, values))
        loss = 0.0
        for outcomes in sequences.values():
            p = params.initial
            for outcome in outcomes:
                prediction = params.predict(p)
                loss -= float(np.log(max(1e-12, prediction if outcome else 1 - prediction)))
                p = params.update(p, outcome)
        return loss
    rng = np.random.default_rng(seed)
    bounds = [(0.01, 0.99), (0.001, 0.4), (0.001, 0.45), (0.001, 0.4)]
    starts = [[0.2, 0.1, 0.2, 0.1]] + [rng.uniform([b[0] for b in bounds], [b[1] for b in bounds]) for _ in range(restarts - 1)]
    fits = [minimize(objective, start, bounds=bounds, method='L-BFGS-B') for start in starts]
    finite = [fit for fit in fits if np.isfinite(fit.fun)]
    if not finite:
        raise RuntimeError('BKT optimization failed')
    best = min(finite, key=lambda fit: fit.fun)
    return (BKTParameters(*map(float, best.x)), {'method': 'bounded_maximum_likelihood', 'negative_log_likelihood': float(best.fun), 'optimizer_converged': bool(best.success), 'optimizer_message': str(best.message), 'restarts': len(fits), 'events': len(rows), 'sequences': len(sequences)})

def load_parameters(path: str | Path | None) -> BKTParameters:
    if path is None:
        return BKTParameters()
    data = json.loads(Path(path).read_text())
    return BKTParameters(**data.get('parameters', data))

class LearnerStore:

    def __init__(self, path: str | Path, params: BKTParameters):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.params = params
        self.db.executescript('\n            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);\n            CREATE TABLE IF NOT EXISTS states (\n                learner TEXT, skill TEXT, mastery REAL, n INTEGER, last_time TEXT,\n                PRIMARY KEY(learner,skill));\n            CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT);\n        ')
        fingerprint = object_digest(asdict(params))
        previous = self.db.execute("SELECT value FROM metadata WHERE key='parameter_hash'").fetchone()
        if previous and previous[0] != fingerprint:
            self.db.close()
            raise ValueError('BKT parameters changed; replay events into a new state database')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('parameter_hash',?)", (fingerprint,))
        self.db.commit()

    def close(self):
        self.db.close()

    def snapshot(self, learner: str) -> dict:
        return {skill: {'mastery': p, 'evidence_count': n, 'last_time': stamp} for skill, p, n, stamp in self.db.execute('SELECT skill,mastery,n,last_time FROM states WHERE learner=?', (learner,))}

    def observe(self, event: dict) -> dict:
        validate_events([event])
        key = (event['learner_id'], event['skill_id'])
        payload = json.dumps(event, sort_keys=True)
        timestamp = datetime.fromisoformat(event['timestamp'])
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            previous = self.db.execute('SELECT payload FROM events WHERE id=?', (event['event_id'],)).fetchone()
            state = self.db.execute('SELECT mastery,n,last_time FROM states WHERE learner=? AND skill=?', key).fetchone()
            if previous:
                if previous[0] != payload:
                    raise ValueError('Event ID reused with a different payload')
                return {**self.snapshot(key[0])[key[1]], 'duplicate': True}
            p, n, last = state or (self.params.initial, 0, None)
            if last and timestamp < datetime.fromisoformat(last):
                raise ValueError('Out-of-order assessment; replay events chronologically')
            updated = self.params.update(p, event['correct'])
            self.db.execute('INSERT OR REPLACE INTO states VALUES (?,?,?,?,?)', (*key, updated, n + 1, event['timestamp']))
            self.db.execute('INSERT INTO events VALUES (?,?)', (event['event_id'], payload))
        return {'mastery': updated, 'evidence_count': n + 1, 'last_time': event['timestamp'], 'duplicate': False}

def gap(state: dict | None, threshold=0.65, min_evidence=2) -> dict:
    if not state or state['evidence_count'] < min_evidence:
        return {'status': 'insufficient_evidence', 'priority': None}
    p = state['mastery']
    return {'status': 'knowledge_gap' if p < threshold else 'no_gap', 'priority': 1 - p if p < threshold else 0.0}
