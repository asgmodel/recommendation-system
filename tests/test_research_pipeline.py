from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json
from rdflib import RDF, URIRef
from hlo_recommender.data import validate_partitions
from hlo_recommender.evaluation import answer_f1, binary_metrics, ranking_metrics
from hlo_recommender.io import config, digest, read_jsonl
from hlo_recommender.knowledge import HLO, template_query
from hlo_recommender.models.bkt import BKTParameters, LearnerStore, gap
from hlo_recommender.models.mt5 import MT5Generator, serialize_input
from hlo_recommender.models.dueling import validate_transitions
from hlo_recommender.pipeline import Runtime
from hlo_recommender.query import QueryRejected, validate
from hlo_recommender.recommendation import eligibility, rank
ROOT = Path(__file__).resolve().parents[1]

class ResearchPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime.from_file(ROOT / 'configs/default.toml')

    def test_source_ontology_is_unchanged(self):
        self.assertEqual(self.runtime.knowledge.ontology_sha256, 'b8e8bad2545126b90e5bc2a6d6fdfb47af4846ee8fda39eaf56326bd0e9aa342')
        self.assertEqual(self.runtime.knowledge.statistics()['classes'], 1642)
        self.assertFalse(list(self.runtime.knowledge.schema.subjects(RDF.type, HLO.LearningResource)))

    def test_catalog_metadata_matches_rdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog=json.loads((ROOT/'data/development/resources.json').read_text())
            catalog['resources'][0]['minutes']+=1
            path=Path(tmp)/'resources.json'
            path.write_text(json.dumps(catalog))
            cfg=config(ROOT/'configs/default.toml')
            cfg['resources']=str(path)
            with self.assertRaises(ValueError):Runtime(cfg)

    def test_schema_selection_does_not_read_reference_answers(self):
        text = self.runtime.grounder.schema_text('Study arterial blood gas interpretation')
        self.assertIn(str(HLO.K002), text)
        self.assertNotIn(str(HLO.K003), text)
        serialized = serialize_input('question', text, {'level': 'Junior', 'sparql': 'SECRET', 'relevance': 'SECRET'})
        self.assertNotIn('SECRET', serialized)

    def test_unrecognized_question_requires_clarification(self):
        result = self.runtime.retrieve('Unmapped topic', {'level': 'Junior', 'minutes': 10}, self.runtime.generator('template'))
        self.assertEqual(result['status'], 'clarification_required')

    def test_time_constraint_changes_retrieval(self):
        query = template_query([str(HLO.K002)], 'Junior', 15)
        result = self.runtime.runner.execute(query)['result']
        shorter = self.runtime.runner.execute(template_query([str(HLO.K002)], 'Junior', 6))['result']
        self.assertGreater(len(result['rows']), len(shorter['rows']))
        self.assertEqual(len(shorter['rows']), 1)

    def test_query_validation_rejects_external_access_and_unknown_identifiers(self):
        queries = ['SELECT ?s WHERE { SERVICE <https://example.org/query> { ?s ?p ?o } }', 'SELECT ?s FROM <https://example.org/graph> WHERE { ?s ?p ?o }', 'DELETE WHERE { ?s ?p ?o }', 'SELECT ?s WHERE { ?s <https://example.org/unknown-property> ?o }']
        for query in queries:
            with self.subTest(query=query), self.assertRaises(QueryRejected):
                validate(query, self.runtime.knowledge.known_terms)

    def test_query_validation_rejects_incompatible_declared_types(self):
        query = f'SELECT ?s WHERE {{ ?s a <{HLO.LearningResource}> ; <{HLO.requiresCompetency}> ?c }}'
        with self.assertRaises(QueryRejected):
            self.runtime.runner.execute(query)

    def test_non_resource_projection_is_rejected(self):
        query = f'SELECT ?s WHERE {{ ?s a <http://www.w3.org/2002/07/owl#Class> }} LIMIT 2'
        self.assertEqual(self.runtime.select([query])['status'], 'no_valid_query')

    def test_missing_model_never_falls_back_to_a_template(self):
        with self.assertRaises(ValueError):
            self.runtime.generator('mt5', ROOT / 'artifacts/missing-checkpoint')

    def test_question_families_are_disjoint(self):
        splits = {name: read_jsonl(ROOT / f'data/development/questions.{name}.jsonl') for name in ('train', 'validation', 'test')}
        self.assertEqual(validate_partitions(splits)['test']['records'], 9)
        bad = {name: list(rows) for name, rows in splits.items()}
        bad['test'].append({**splits['train'][0], 'id': 'new-id'})
        with self.assertRaises(ValueError):
            validate_partitions(bad)

    def test_synthetic_training_requires_explicit_permission(self):
        splits = {name: read_jsonl(ROOT / f'data/development/questions.{name}.jsonl') for name in ('train', 'validation')}
        with self.assertRaises(ValueError):
            validate_partitions(splits, require_reviewed=True)

    def test_bkt_updates_follow_assessment_evidence(self):
        p = BKTParameters()
        self.assertGreater(p.update(0.2, True), 0.2)
        self.assertLess(p.update(0.2, False), 0.2)
        self.assertEqual(gap({'mastery': 0.1, 'evidence_count': 1})['status'], 'insufficient_evidence')

    def test_event_updates_are_idempotent_and_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LearnerStore(Path(tmp) / 'state.sqlite', BKTParameters())
            event = {'event_id': 'x1', 'learner_id': 'l1', 'skill_id': str(HLO.K002), 'correct': False, 'timestamp': '2025-01-02T00:00:00+00:00', 'data_status': 'synthetic'}
            first = store.observe(event)
            second = store.observe(event)
            self.assertEqual(first['mastery'], second['mastery'])
            self.assertTrue(second['duplicate'])
            with self.assertRaises(ValueError):
                store.observe({**event, 'correct': True})
            with self.assertRaises(ValueError):
                store.observe({**event, 'event_id': 'earlier', 'timestamp': '2025-01-01T00:00:00+00:00'})
            store.close()
            with self.assertRaises(ValueError):
                LearnerStore(Path(tmp) / 'state.sqlite', BKTParameters(learn=0.2))

    def test_eligibility_precedes_ranking(self):
        resource = next((r for r in self.runtime.resources if r.id.endswith('K002-6')))
        self.assertIn('duration', eligibility(resource, {'level': 'Junior', 'minutes': 15}, [str(HLO.K002)], {}))
        advanced = next((r for r in self.runtime.resources if r.id.endswith('K002-3')))
        reasons = eligibility(advanced, {'level': 'Senior', 'minutes': 30}, [str(HLO.K002)], {})
        self.assertIn('prerequisite_unobserved', reasons)

    def test_adaptive_ranking_changes_with_mastery(self):
        pool = [r for r in self.runtime.resources if str(HLO.K002) in r.skills]
        kwargs = {'context': {'level': 'Junior', 'minutes': 15}, 'target_skills': [str(HLO.K002)], 'weights': self.runtime.config['weights']}
        low = rank(pool, mastery={str(HLO.K002): 0.1}, **kwargs)
        high = rank(pool, mastery={str(HLO.K002): 0.99}, **kwargs)
        self.assertNotEqual(low['ranked'][0]['resource_id'], high['ranked'][0]['resource_id'])
        static_low = rank(pool, mastery={str(HLO.K002): 0.1}, adaptive=False, **kwargs)
        static_high = rank(pool, mastery={str(HLO.K002): 0.99}, adaptive=False, **kwargs)
        self.assertEqual(static_low['ranked'][0]['resource_id'], static_high['ranked'][0]['resource_id'])

    def test_execution_evaluation_handles_projection_and_empty_answers(self):
        gold = {'kind': 'SELECT', 'variables': ['resource'], 'rows': [['<urn:r>']], 'truncated': False}
        same = {**gold, 'variables': ['other']}
        self.assertEqual(answer_f1(same, gold), 1.0)
        self.assertEqual(answer_f1({**gold, 'variables': ['x', 'y'], 'rows': [['<urn:r>', None]]}, gold), 0.0)
        self.assertEqual(answer_f1({**gold, 'rows': []}, gold), 0.0)

    def test_predictive_and_ranking_metrics(self):
        scores = binary_metrics([False, True], [0.5, 0.5])
        self.assertAlmostEqual(scores['auc'], 0.5)
        self.assertAlmostEqual(scores['brier'], 0.25)
        best = ranking_metrics(['a', 'b'], {'a': 3, 'b': 1})
        reversed_ = ranking_metrics(['b', 'a'], {'a': 3, 'b': 1})
        self.assertAlmostEqual(best['ndcg_at_k'], 1.0)
        self.assertGreater(best['ndcg_at_k'], reversed_['ndcg_at_k'])

    def test_invalid_policy_transitions_are_rejected_without_torch(self):
        row = {'episode_id': 'e1', 'step': 0, 'state': [0.2, 0.8, 0.5, 0.0], 'action': 1, 'reward': 0.1, 'next_state': [0.3, 0.7, 0.4, 0.0], 'done': False, 'allowed_actions': [True, False, True], 'next_allowed_actions': [True, True, True], 'data_status': 'synthetic'}
        with self.assertRaises(ValueError):
            validate_transitions([row], 4, 3)
if __name__ == '__main__':
    unittest.main()
