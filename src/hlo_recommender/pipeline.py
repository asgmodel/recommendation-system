"""Question grounding, query execution and resource recommendation."""
from __future__ import annotations
from rdflib import URIRef, RDF
from rdflib.util import from_n3
from .io import config, resolve
from .knowledge import Grounder, Knowledge, template_query, HLO, EXT, LEVELS
from .models.bkt import gap
from .models.mt5 import MT5Generator
from .query import QueryRejected, QueryRunner
from .recommendation import load_resources, rank

class TemplateGenerator:
    name = 'template'

    def __init__(self, grounder):
        self.grounder = grounder

    def generate(self, question, schema, context):
        return [template_query(self.grounder.skills(question), context['level'], context['minutes'])]

class Runtime:

    def __init__(self, cfg: dict):
        self.config = cfg
        self.knowledge = Knowledge.load(resolve(cfg, cfg['ontology']), resolve(cfg, cfg['catalog']))
        self.grounder = Grounder(self.knowledge, resolve(cfg, cfg['aliases']))
        self.resources, self.data_status = load_resources(resolve(cfg, cfg['resources']))
        self.by_id = {r.id: r for r in self.resources}
        self.runner = QueryRunner(self.knowledge, float(cfg.get('query_timeout', 10)), int(cfg.get('max_query_rows', 1000)))
        for resource in self.resources:
            if (URIRef(resource.id), None, None) not in self.knowledge.graph:
                raise ValueError('Resource metadata has no RDF record: ' + resource.id)
            for skill in resource.skills:
                if URIRef(skill) not in self.knowledge.known_terms:
                    raise ValueError('Unknown catalog skill: ' + skill)
            node = URIRef(resource.id)
            durations = list(self.knowledge.graph.objects(node, EXT.estimatedDurationMinutes))
            if len(durations) != 1 or durations[0].toPython() != resource.minutes:
                raise ValueError('RDF and metadata durations differ: ' + resource.id)
            linked_skills = set()
            for objective in self.knowledge.graph.objects(node, HLO.supportsObjective):
                for competency in self.knowledge.graph.objects(objective, HLO.requiresCompetency):
                    for instance in self.knowledge.graph.objects(competency, HLO.developsSkill):
                        for typ in self.knowledge.graph.objects(instance, RDF.type):
                            if typ != HLO.ClinicalSkill and HLO.ClinicalSkill in self.knowledge.ancestors(typ):
                                linked_skills.add(str(typ))
            if linked_skills != set(resource.skills):
                raise ValueError('RDF and metadata skills differ: ' + resource.id)
            linked_levels = {level for value in self.knowledge.graph.objects(node, HLO.recommendedFor)
                             for level in LEVELS if (value, RDF.type, HLO[level]) in self.knowledge.graph}
            if linked_levels != set(resource.levels):
                raise ValueError('RDF and metadata levels differ: ' + resource.id)
            if any(URIRef(skill) not in self.knowledge.known_terms for skill in resource.prerequisites):
                raise ValueError('Unknown prerequisite skill: ' + resource.id)

    @classmethod
    def from_file(cls, path):
        return cls(config(path))

    def generator(self, name, checkpoint=None):
        if name == 'template':
            return TemplateGenerator(self.grounder)
        if name == 'mt5':
            if checkpoint is None:
                raise ValueError('mT5 inference requires --checkpoint')
            return MT5Generator(checkpoint, self.knowledge.ontology_sha256, self.grounder.sha256)
        raise ValueError('Unsupported generator: ' + name)

    def select(self, candidates):
        errors = []
        for candidate in candidates:
            try:
                execution = self.runner.execute(candidate)
                result = execution['result']
                if result['kind'] != 'SELECT' or len(result['variables']) != 1:
                    raise QueryRejected('Resource queries must project one SELECT column')
                ids = []
                for row in result['rows']:
                    term = from_n3(row[0]) if row[0] is not None else None
                    if not isinstance(term, URIRef) or str(term) not in self.by_id:
                        raise QueryRejected('Query projected a non-catalog resource')
                    ids.append(str(term))
                return {'status': 'executed', 'query': candidate, 'resource_ids': sorted(set(ids)), 'execution': execution, 'candidate_errors': errors}
            except (QueryRejected, ValueError) as exc:
                errors.append(str(exc))
        return {'status': 'no_valid_query', 'query': None, 'resource_ids': [], 'candidate_errors': errors}

    def retrieve(self, question, context, generator):
        if context.get('level') not in LEVELS or type(context.get('minutes')) is not int or context['minutes'] <= 0:
            raise ValueError('A supported level and positive integer time budget are required')
        skills = self.grounder.skills(question)
        if not skills:
            return {'status': 'clarification_required', 'query': None, 'resource_ids': [], 'target_skills': []}
        schema = self.grounder.schema_text(question)
        candidates = generator.generate(question, schema, context)
        return {**self.select(candidates), 'target_skills': skills, 'generator': generator.name}

    def recommend(self, question, context, generator, *, state=None, adaptive=True):
        retrieved = self.retrieve(question, context, generator)
        states = state or {}
        mastery = {key: value['mastery'] if isinstance(value, dict) else value for key, value in states.items()}
        if retrieved['status'] != 'executed':
            return {**retrieved, 'ranked': [], 'data_status': self.data_status}
        pool = [self.by_id[key] for key in retrieved['resource_ids']]
        ranked = rank(pool, context, retrieved['target_skills'], mastery, self.config['weights'], adaptive=adaptive, top_k=self.config['top_k'], initial_mastery=self.config['bkt']['initial'])
        return {**retrieved, **ranked, 'data_status': self.data_status, 'gaps': {key: gap(value) for key, value in states.items() if isinstance(value, dict)}, 'ontology_sha256': self.knowledge.ontology_sha256, 'catalog_sha256': self.knowledge.catalog_sha256}
