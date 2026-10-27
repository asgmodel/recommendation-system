"""Deterministic synthetic data for development and integration experiments."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from rdflib import Graph, Literal, RDF, RDFS, OWL, XSD, URIRef
from .io import grouped_split, write_json, write_jsonl
from .knowledge import HLO, EXT, LEVELS, template_query
SKILLS = {'K001': 'Clinical Order Entry', 'K002': 'Arterial Blood Gas Interpretation', 'K003': 'Central Line Insertion', 'K004': 'Trauma Assessment', 'K005': 'Point of Care Ultrasound'}

def create_development_data(root: str | Path, seed=42) -> dict:
    root = Path(root)
    rng = random.Random(seed)
    target = root / 'data/development'
    target.mkdir(parents=True, exist_ok=True)
    aliases = {label.lower(): str(HLO[key]) for key, label in SKILLS.items()}
    aliases.update({'abg': str(HLO.K002), 'pocus': str(HLO.K005)})
    write_json(root / 'configs/aliases.json', aliases)
    graph = Graph()
    graph.bind('hlo', HLO)
    graph.bind('ext', EXT)
    graph.add((EXT.estimatedDurationMinutes, RDF.type, OWL.DatatypeProperty))
    graph.add((EXT.estimatedDurationMinutes, RDFS.domain, HLO.LearningResource))
    graph.add((EXT.estimatedDurationMinutes, RDFS.range, XSD.integer))
    for level in LEVELS:
        graph.add((EXT['level/' + level], RDF.type, HLO[level]))
        graph.add((EXT['level/' + level], RDF.type, HLO.ResidentLevel))
    resources = []
    variants = [('Primer', 6, 0.2, 0.78, LEVELS, ()), ('Worked Case', 12, 0.55, 0.88, LEVELS, ()), ('Advanced Case', 18, 0.85, 0.98, ('Senior', 'Fellow'), ('self',)), ('Extended Case', 12, 0.8, 0.96, ('Junior',), ()), ('Short Review', 8, 0.35, 0.84, LEVELS, ()), ('Reference Chapter', 45, 0.5, 0.99, LEVELS, ())]
    for code, label in SKILLS.items():
        skill = EXT['skill/' + code]
        objective = EXT['objective/' + code]
        competency = EXT['competency/' + code]
        for typ in (HLO[code], HLO.ClinicalSkill):
            graph.add((skill, RDF.type, typ))
        graph.add((objective, RDF.type, HLO.LearningObjective))
        graph.add((competency, RDF.type, HLO.Competency))
        graph.add((objective, HLO.requiresCompetency, competency))
        graph.add((competency, HLO.developsSkill, skill))
        for index, (kind, minutes, difficulty, quality, levels, pre) in enumerate(variants):
            identifier = str(EXT[f'resource/{code}-{index + 1}'])
            row = {'id': identifier, 'title': f'{label} {kind}', 'skills': [str(HLO[code])], 'minutes': minutes, 'levels': list(levels), 'quality': quality, 'difficulty': difficulty, 'activity': 'Any', 'location': 'Any', 'prerequisites': [str(HLO[code])] if pre else [], 'active': True}
            resources.append(row)
            node = URIRef(identifier)
            graph.add((node, RDF.type, HLO.LearningResource))
            graph.add((node, RDFS.label, Literal(row['title'], lang='en')))
            graph.add((node, HLO.supportsObjective, objective))
            graph.add((node, EXT.estimatedDurationMinutes, Literal(minutes, datatype=XSD.integer)))
            for level in levels:
                graph.add((node, HLO.recommendedFor, EXT['level/' + level]))
    write_json(target / 'resources.json', {'data_status': 'synthetic', 'resources': resources})
    graph.serialize(target / 'catalog.ttl', format='turtle')
    questions = []
    for code, label in SKILLS.items():
        for index, level in enumerate(LEVELS):
            minutes = (15, 25, 30)[index]
            context = {'level': level, 'minutes': minutes, 'location': 'Hospital', 'activity': 'Study'}
            phrasings = [f'Find resources for {label} for a {level} learner within {minutes} minutes.', f'I have {minutes} minutes at {level} level to study {label}.', f'Which {label} learning resources suit {level} level and {minutes} minutes?']
            for j, question in enumerate(phrasings):
                mastery = {str(HLO[code]): (0.12, 0.55, 0.85)[j]}
                grades = {}
                for resource in resources:
                    if str(HLO[code]) not in resource['skills'] or level not in resource['levels'] or resource['minutes'] > minutes:
                        continue
                    if resource['prerequisites'] and mastery[str(HLO[code])] < 0.65:
                        continue
                    distance = abs(resource['difficulty'] - (0.15 + 0.75 * mastery[str(HLO[code])]))
                    grades[resource['id']] = 3 if distance < 0.16 else 1 if distance < 0.4 else 0
                questions.append({'id': f'{code}-{level}-{j + 1}', 'family_id': f'{code}-{level}', 'data_status': 'synthetic', 'question': question, 'context': context, 'sparql': template_query([str(HLO[code])], level, minutes), 'learner_state': mastery, 'relevance': grades, 'reference_kind': 'synthetic_rule'})
    split = grouped_split(questions, 'family_id', seed)
    for name, rows in split.items():
        write_jsonl(target / f'questions.{name}.jsonl', rows)
    events = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for learner in range(40):
        for code in SKILLS:
            mastered = rng.random() < 0.25
            for step in range(8):
                correct = rng.random() < (0.92 if mastered else 0.22)
                events.append({'event_id': f'a-{learner:03d}-{code}-{step}', 'learner_id': f'learner-{learner:03d}', 'skill_id': str(HLO[code]), 'correct': correct, 'timestamp': (start + timedelta(days=step, minutes=learner)).isoformat(), 'data_status': 'synthetic'})
                if not mastered and rng.random() < 0.12:
                    mastered = True
    event_split = grouped_split(events, 'learner_id', seed)
    for name, rows in event_split.items():
        write_jsonl(target / f'assessments.{name}.jsonl', rows)
    manifest = {'data_status': 'synthetic', 'purpose': 'development_and_integration', 'seed': seed, 'question_records': len(questions), 'question_families': 15, 'resource_records': len(resources), 'assessment_records': len(events), 'references': 'deterministic_synthetic_policy', 'question_splits': {k: len(v) for k, v in split.items()}, 'assessment_splits': {k: len(v) for k, v in event_split.items()}}
    write_json(target / 'manifest.json', manifest)
    return manifest
