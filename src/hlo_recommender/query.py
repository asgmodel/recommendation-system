"""Structural validation and bounded local RDF execution.

This validator checks syntax, terms and supported operations. It does not prove
that a query captures the user's meaning; independent evaluation is required.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from time import perf_counter
import multiprocessing as mp
from queue import Empty
from typing import Any
from rdflib import Graph, URIRef, Variable, RDF, RDFS, OWL, XSD
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from .knowledge import Knowledge

class QueryRejected(ValueError):
    pass

def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from walk(child)
    elif hasattr(value, 'args'):
        yield from walk(value.args)
    elif hasattr(value, 'arg'):
        yield from walk(value.arg)
    elif hasattr(value, 'path'):
        yield from walk(value.path)

def validate(query: str, known_terms: set[URIRef]) -> dict:
    if len(query) > 20000:
        raise QueryRejected('query_too_long')
    try:
        parsed = parseQuery(query)
        algebra = translateQuery(parsed).algebra
    except Exception as exc:
        raise QueryRejected('invalid_sparql') from exc
    if algebra.name not in {'SelectQuery', 'AskQuery'}:
        raise QueryRejected('unsupported_query_form')
    if any((isinstance(node, CompValue) and node.name in {'ServiceGraphPattern', 'Graph'} for node in walk(algebra))):
        raise QueryRejected('external_or_named_graph_not_supported')
    if algebra.get('datasetClause'):
        raise QueryRejected('dataset_clause_not_supported')
    standard_terms = {RDF.type, RDFS.subClassOf, RDFS.label, RDFS.domain, RDFS.range, OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, XSD.integer, XSD.decimal, XSD.string, XSD.boolean}
    unknown = sorted({str(x) for x in walk(algebra) if isinstance(x, URIRef) and x not in known_terms | standard_terms})
    if unknown:
        raise QueryRejected('unknown_identifiers: ' + ', '.join(unknown[:5]))
    return {'syntax_valid': True, 'known_identifiers': True, 'query_form': algebra.name, 'semantic_correctness': 'not_proven'}

def _execute_worker(ontology: str, catalog: str, query: str, limit: int, output):
    try:
        graph = Graph().parse(ontology, format='turtle' if ontology.endswith('.ttl') else 'xml')
        graph.parse(catalog, format='turtle')
        result = graph.query(query)
        if result.type == 'ASK':
            output.put({'kind': 'ASK', 'value': bool(result.askAnswer), 'truncated': False})
            return
        rows = []
        truncated = False
        for row in result:
            if len(rows) >= limit:
                truncated = True
                break
            rows.append([term.n3() if term is not None else None for term in row])
        output.put({'kind': 'SELECT', 'variables': [str(v) for v in result.vars], 'rows': rows, 'truncated': truncated})
    except Exception as exc:
        output.put({'error': type(exc).__name__})

@dataclass
class QueryRunner:
    knowledge: Knowledge
    timeout_seconds: float = 10.0
    max_rows: int = 1000
    _cache: dict = field(default_factory=dict, repr=False)

    def _type_checks(self, query: str) -> None:
        algebra = translateQuery(parseQuery(query)).algebra
        triples = [triple for node in walk(algebra) if isinstance(node, CompValue) and node.name == 'BGP' for triple in node['triples']]
        types = {}
        for subject, predicate, obj in triples:
            if predicate == RDF.type and isinstance(obj, URIRef):
                types.setdefault(subject, set()).add(obj)
        for subject, predicate, obj in triples:
            if not isinstance(predicate, URIRef):
                continue
            for term, relation in ((subject, RDFS.domain), (obj, RDFS.range)):
                expected = self.knowledge.graph.value(predicate, relation)
                declared = types.get(term, set())
                if expected and declared and (not any((expected in self.knowledge.ancestors(actual) for actual in declared))):
                    raise QueryRejected('incompatible_declared_type: ' + str(predicate))

    def execute(self, query: str) -> dict:
        if query in self._cache:
            result = deepcopy(self._cache[query])
            result['cache_hit'] = True
            return result
        started = perf_counter()
        checks = validate(query, self.knowledge.known_terms)
        self._type_checks(query)
        ctx = mp.get_context('spawn')
        output = ctx.Queue()
        process = ctx.Process(target=_execute_worker, args=(str(self.knowledge.ontology), str(self.knowledge.catalog), query, self.max_rows, output))
        process.start()
        try:
            result = output.get(timeout=self.timeout_seconds)
        except Empty as exc:
            raise QueryRejected('execution_timeout_or_worker_failure') from exc
        finally:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join()
            output.close()
            output.join_thread()
        if 'error' in result:
            raise QueryRejected('execution_error: ' + result['error'])
        if result.get('truncated'):
            raise QueryRejected('result_limit_exceeded')
        record = {'checks': checks, 'result': result, 'ontology_sha256': self.knowledge.ontology_sha256, 'catalog_sha256': self.knowledge.catalog_sha256, 'elapsed_seconds': perf_counter() - started, 'cache_hit': False}
        self._cache[query] = deepcopy(record)
        return record
