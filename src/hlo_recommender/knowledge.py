"""Versioned HLO schema and separately loaded resource catalog."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef
from .io import digest, object_digest
HLO = Namespace('http://example.org/hlo.owl#')
EXT = Namespace('https://example.org/hlo-research/')
LEVELS = ('Junior', 'Senior', 'Fellow')

@dataclass
class Knowledge:
    ontology: Path
    catalog: Path
    graph: Graph
    schema: Graph
    ontology_sha256: str
    catalog_sha256: str
    known_terms: set

    @classmethod
    def load(cls, ontology: str | Path, catalog: str | Path) -> 'Knowledge':
        ontology, catalog = (Path(ontology).resolve(), Path(catalog).resolve())
        schema = Graph().parse(ontology, format='xml')
        operational = Graph().parse(catalog, format='turtle')
        graph = Graph()
        graph += schema
        graph += operational
        known = {v for triple in graph for v in triple if isinstance(v, URIRef)}
        return cls(ontology, catalog, graph, schema, digest(ontology), digest(catalog), known)

    def statistics(self) -> dict:
        return {'ontology_sha256': self.ontology_sha256, 'catalog_sha256': self.catalog_sha256, 'ontology_triples': len(self.schema), 'combined_triples': len(self.graph), 'classes': len(set(self.schema.subjects(RDF.type, OWL.Class))), 'object_properties': len(set(self.schema.subjects(RDF.type, OWL.ObjectProperty))), 'datatype_properties': len(set(self.schema.subjects(RDF.type, OWL.DatatypeProperty))), 'resource_instances': len(set(self.graph.subjects(RDF.type, HLO.LearningResource)))}

    def ancestors(self, term: URIRef) -> set:
        seen, pending = ({term}, [term])
        while pending:
            for parent in self.graph.objects(pending.pop(), RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent not in seen:
                    seen.add(parent)
                    pending.append(parent)
        return seen

class Grounder:

    def __init__(self, knowledge: Knowledge, aliases: str | Path):
        self.knowledge = knowledge
        self.aliases = json.loads(Path(aliases).read_text())
        self.sha256 = object_digest(self.aliases)
        for iri in self.aliases.values():
            if (URIRef(iri), RDF.type, OWL.Class) not in knowledge.schema:
                raise ValueError(f'Alias points to an unknown class: {iri}')

    def skills(self, question: str) -> list[str]:
        question = question.casefold()
        return sorted({iri for alias, iri in self.aliases.items() if alias.casefold() in question})

    def schema_text(self, question: str) -> str:
        skills = self.skills(question)
        if not skills:
            raise ValueError('No supported skill identified; clarification is required')
        classes = [HLO.LearningResource, HLO.LearningObjective, HLO.Competency, HLO.ClinicalSkill, HLO.ResidentLevel, HLO.Junior, HLO.Senior, HLO.Fellow]
        props = [HLO.supportsObjective, HLO.requiresCompetency, HLO.developsSkill, HLO.recommendedFor, EXT.estimatedDurationMinutes]
        lines = ['OUTPUT SELECT DISTINCT ?resource', 'PREFIX hlo: <' + str(HLO) + '>', 'PREFIX ext: <' + str(EXT) + '>']
        for prop in props:
            lines.append(f'PROPERTY <{prop}> DOMAIN <{self.knowledge.graph.value(prop, RDFS.domain)}> RANGE <{self.knowledge.graph.value(prop, RDFS.range)}>')
        for term in classes + [URIRef(x) for x in skills]:
            label = self.knowledge.schema.value(term, RDFS.label) or str(term).split('#')[-1]
            parents = sorted((str(x) for x in self.knowledge.schema.objects(term, RDFS.subClassOf)))
            lines.append(f'CLASS <{term}> LABEL {label} PARENTS {json.dumps(parents)}')
        return '\n'.join(lines)

def template_query(skills: list[str], level: str, minutes: int) -> str:
    if not skills or level not in LEVELS or type(minutes) is not int or (minutes <= 0):
        raise ValueError('A supported skill, level and positive integer time budget are required')
    targets = ' '.join((f'<{URIRef(skill)}>' for skill in skills))
    return f'PREFIX hlo: <{HLO}>\nPREFIX ext: <{EXT}>\nSELECT DISTINCT ?resource WHERE {{\n  VALUES ?skillClass {{ {targets} }}\n  ?resource a hlo:LearningResource ;\n    hlo:supportsObjective ?objective ;\n    hlo:recommendedFor ?level ; ext:estimatedDurationMinutes ?duration .\n  ?objective hlo:requiresCompetency ?competency .\n  ?competency hlo:developsSkill ?skill .\n  ?skill a ?skillClass .\n  ?level a hlo:{level} .\n  FILTER (?duration <= {minutes})\n}} ORDER BY ?resource'
