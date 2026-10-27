"""Execution, ranking and predictive metrics."""
from __future__ import annotations
import math
from collections import Counter
from .query import QueryRejected

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None

def answer_set(result: dict) -> set:
    if result.get('truncated'):
        raise ValueError('Truncated results cannot be evaluated')
    if result['kind'] == 'ASK':
        return {(bool(result['value']),)}
    return {tuple(row) for row in result['rows']}

def answer_f1(predicted: dict, gold: dict) -> float:
    if predicted['kind'] != gold['kind']:
        return 0.0
    if gold['kind'] == 'SELECT' and len(predicted['variables']) != len(gold['variables']):
        return 0.0
    p, g = (answer_set(predicted), answer_set(gold))
    if not p and (not g):
        return 1.0
    return 2 * len(p & g) / (len(p) + len(g)) if p or g else 0.0

def evaluate_queries(rows, predictions, runner):
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Nonempty matching cases and predictions are required')
    records = []
    for row, prediction in zip(rows, predictions):
        gold = runner.execute(row['sparql'])['result']
        empty = gold['kind'] == 'SELECT' and (not gold['rows'])
        try:
            candidate = runner.execute(prediction)
            f1 = answer_f1(candidate['result'], gold)
            valid, error = (True, None)
        except (QueryRejected, ValueError) as exc:
            valid, f1, error = (False, 0.0, str(exc))
        records.append({'id': row['id'], 'family_id': row['family_id'], 'answer_f1': f1, 'execution_valid': valid, 'empty_reference': empty, 'error': error})
    return {'n': len(records), 'answer_f1': mean((r['answer_f1'] for r in records)), 'nonempty_f1': mean((r['answer_f1'] for r in records if not r['empty_reference'])), 'empty_f1': mean((r['answer_f1'] for r in records if r['empty_reference'])), 'valid_rate': mean((r['execution_valid'] for r in records)), 'records': records}

def ranking_metrics(ids: list[str], grades: dict[str, int], k=5) -> dict:
    if k <= 0 or len(ids) != len(set(ids)):
        raise ValueError('Invalid ranking or cutoff')
    if any((not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in grades.values())):
        raise ValueError('Invalid relevance grades')
    relevant = {key for key, value in grades.items() if value > 0}
    top = ids[:k]
    hits = [int(key in relevant) for key in top]
    ranks = [i + 1 for i, value in enumerate(hits) if value]
    dcg = sum(((2 ** grades.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(top)))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(ideal)))
    precision_sum = sum((sum(hits[:i + 1]) / (i + 1) for i, hit in enumerate(hits) if hit))
    return {'top1': float(bool(ids) and ids[0] in relevant), 'hit3': float(any((x in relevant for x in ids[:3]))), 'hit5': float(any((x in relevant for x in ids[:5]))), 'mrr': 1 / ranks[0] if ranks else 0.0, 'map_at_k': precision_sum / min(len(relevant), k) if relevant else 0.0, 'ndcg_at_k': dcg / idcg if idcg else 0.0, 'precision_at_k': sum(hits) / k, 'reference_nonempty': bool(relevant)}

def binary_metrics(labels: list[bool], probabilities: list[float]) -> dict:
    if not labels or len(labels) != len(probabilities):
        raise ValueError('Nonempty aligned predictions are required')
    if any((not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
        raise ValueError('Invalid probabilities')
    eps = 1e-12
    log_loss = -mean((int(y) * math.log(max(eps, p)) + (1 - int(y)) * math.log(max(eps, 1 - p)) for y, p in zip(labels, probabilities)))
    order = sorted(zip(probabilities, labels))
    positive_ranks, i = (0.0, 0)
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        positive_ranks += average_rank * sum((int(x[1]) for x in order[i:j]))
        i = j
    positives = sum(labels)
    negatives = len(labels) - positives
    auc = (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives) if positives and negatives else None
    bins = []
    for b in range(10):
        selected = [(int(y), p) for y, p in zip(labels, probabilities) if min(9, int(p * 10)) == b]
        if selected:
            bins.append({'bin': b, 'n': len(selected), 'predicted': mean((p for y, p in selected)), 'observed': mean((y for y, p in selected))})
    return {'n': len(labels), 'log_loss': log_loss, 'brier': mean(((int(y) - p) ** 2 for y, p in zip(labels, probabilities))), 'accuracy': mean(((p >= 0.5) == bool(y) for y, p in zip(labels, probabilities))), 'auc': auc, 'calibration': bins}
