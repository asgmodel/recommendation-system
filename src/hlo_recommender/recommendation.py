"""Resource eligibility and explicit static/adaptive ranking."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from .knowledge import LEVELS

@dataclass(frozen=True)
class Resource:
    id: str
    title: str
    skills: tuple[str, ...]
    minutes: int
    levels: tuple[str, ...]
    quality: float
    difficulty: float
    activity: str = 'Any'
    location: str = 'Any'
    prerequisites: tuple[str, ...] = ()
    active: bool = True

    def __post_init__(self):
        if type(self.minutes) is not int or self.minutes <= 0:
            raise ValueError('Resource duration must be a positive integer')
        if not self.skills or not self.levels or any((x not in LEVELS for x in self.levels)):
            raise ValueError('Resource skill and level metadata are required')
        if any((not math.isfinite(x) or not 0 <= x <= 1 for x in (self.quality, self.difficulty))):
            raise ValueError('Quality and difficulty must be finite values in [0, 1]')

def load_resources(path: str | Path) -> tuple[list[Resource], str]:
    data = json.loads(Path(path).read_text())
    rows = []
    for row in data['resources']:
        row = dict(row)
        for key in ('skills', 'levels', 'prerequisites'):
            row[key] = tuple(row.get(key, []))
        rows.append(Resource(**row))
    if len({x.id for x in rows}) != len(rows):
        raise ValueError('Duplicate catalog resource IDs')
    return (rows, data['data_status'])

def eligibility(resource: Resource, context: dict, target_skills: list[str], mastery: dict, prerequisite_threshold=0.65) -> list[str]:
    if context.get('level') not in LEVELS or type(context.get('minutes')) is not int or context['minutes'] <= 0:
        raise ValueError('Invalid learner context')
    reasons = []
    if not resource.active:
        reasons.append('inactive')
    if context['level'] not in resource.levels:
        reasons.append('level')
    if resource.minutes > context['minutes']:
        reasons.append('duration')
    if not set(resource.skills) & set(target_skills):
        reasons.append('target_skill')
    for skill in resource.prerequisites:
        if skill not in mastery:
            reasons.append('prerequisite_unobserved')
        elif mastery[skill] < prerequisite_threshold:
            reasons.append('prerequisite_gap')
    return reasons

def rank(resources: list[Resource], context: dict, target_skills: list[str], mastery: dict, weights: dict, *, adaptive=True, top_k=5, initial_mastery=0.2) -> dict:
    if not target_skills or top_k <= 0:
        raise ValueError('Target skills and a positive top_k are required')
    if any((not isinstance(v, (int, float)) or not math.isfinite(v) or (not 0 <= v <= 1) for v in mastery.values())):
        raise ValueError('Invalid mastery probabilities')
    allowed = {'relevance', 'quality', 'time', 'context', 'gap', 'readiness'}
    if set(weights) != allowed or any((not math.isfinite(v) or v < 0 for v in weights.values())):
        raise ValueError('Invalid ranking weights')
    active = {k: v for k, v in weights.items() if adaptive or k not in {'gap', 'readiness'}}
    if sum(active.values()) <= 0:
        raise ValueError('Ranking weights must sum to a positive value')
    active = {k: v / sum(active.values()) for k, v in active.items()}
    ranked, rejected = ([], [])
    for resource in resources:
        reasons = eligibility(resource, context, target_skills, mastery)
        if reasons:
            rejected.append({'resource_id': resource.id, 'reasons': reasons})
            continue
        covered = set(resource.skills) & set(target_skills)
        p = sum((mastery.get(skill, initial_mastery) for skill in covered)) / len(covered)
        expected_difficulty = 0.15 + 0.75 * p
        location = resource.location in ('Any', context.get('location', 'Any'))
        activity = resource.activity in ('Any', context.get('activity', 'Any'))
        features = {'relevance': len(covered) / len(set(target_skills)), 'quality': resource.quality, 'time': 1 - resource.minutes / context['minutes'], 'context': (location + activity) / 2, 'gap': 1 - p, 'readiness': 1 - abs(resource.difficulty - expected_difficulty)}
        score = sum((active[k] * features[k] for k in active))
        ranked.append({'resource_id': resource.id, 'title': resource.title, 'score': score, 'features': features, 'covered_skills': sorted(covered), 'explanation': {'eligibility': 'passed', 'adaptive': adaptive, 'mastery_used': p, 'weights': active}})
    ranked.sort(key=lambda x: (-x['score'], x['resource_id']))
    return {'status': 'ok' if ranked else 'no_eligible_resource', 'ranked': ranked[:top_k], 'eligible_count': len(ranked), 'rejected': rejected}
