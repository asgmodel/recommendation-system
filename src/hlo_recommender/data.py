"""Question and assessment dataset contracts."""
from __future__ import annotations
from datetime import datetime
from .io import object_digest

def validate_questions(rows: list[dict], *, require_reviewed: bool=False) -> None:
    if not rows:
        raise ValueError('Question dataset is empty')
    ids = []
    for row in rows:
        for field in ('id', 'family_id', 'question', 'sparql', 'context', 'data_status'):
            if field not in row:
                raise ValueError(f'Missing question field: {field}')
        if any((not isinstance(row[key], str) or not row[key].strip() for key in ('id', 'family_id', 'question', 'sparql'))):
            raise ValueError('Question identifiers, text and SPARQL must be nonempty strings')
        if not isinstance(row['context'], dict):
            raise ValueError('Question context must be an object')
        ids.append(row['id'])
        if row['data_status'] not in {'synthetic', 'reviewed'}:
            raise ValueError('Unsupported data_status')
        if require_reviewed and row['data_status'] != 'reviewed':
            raise ValueError('Research training requires reviewed data; use --allow-synthetic for development')
        context = row['context']
        if context.get('level') not in {'Junior', 'Senior', 'Fellow'}:
            raise ValueError('Unsupported learner level')
        if type(context.get('minutes')) is not int or context['minutes'] <= 0:
            raise ValueError('Invalid time budget')
        if any((key in context for key in ('sparql', 'gold_sparql', 'relevance', 'schema'))):
            raise ValueError('Reference information must not enter model context')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs')

def validate_partitions(partitions: dict[str, list[dict]], *, require_reviewed=False) -> dict:
    seen_ids, seen_families, seen_queries, seen_questions = (set(), set(), set(), set())
    for name, rows in partitions.items():
        validate_questions(rows, require_reviewed=require_reviewed)
        ids = {row['id'] for row in rows}
        families = {row['family_id'] for row in rows}
        queries = {object_digest(' '.join(row['sparql'].split())) for row in rows}
        questions = {' '.join(row['question'].casefold().split()) for row in rows}
        for label, values, seen in (('ID', ids, seen_ids), ('family', families, seen_families), ('query', queries, seen_queries), ('question', questions, seen_questions)):
            if values & seen:
                raise ValueError(f'{label} leakage involving {name}')
            seen.update(values)
    return {name: {'records': len(rows), 'families': len({r['family_id'] for r in rows})} for name, rows in partitions.items()}

def validate_events(rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Assessment dataset is empty')
    ids = set()
    for row in rows:
        for key in ('event_id', 'learner_id', 'skill_id', 'correct', 'timestamp', 'data_status'):
            if key not in row:
                raise ValueError(f'Missing assessment field: {key}')
        if any((not isinstance(row[key], str) or not row[key].strip() for key in ('event_id', 'learner_id', 'skill_id', 'timestamp'))):
            raise ValueError('Assessment identifiers and timestamps must be nonempty strings')
        if row['data_status'] not in {'synthetic', 'reviewed'}:
            raise ValueError('Unsupported assessment data_status')
        if type(row['correct']) is not bool:
            raise ValueError('correct must be a JSON boolean')
        if row['event_id'] in ids:
            raise ValueError('Duplicate assessment event ID')
        ids.add(row['event_id'])
        stamp = datetime.fromisoformat(row['timestamp'])
        if stamp.tzinfo is None:
            raise ValueError('Assessment timestamps must include a timezone')
