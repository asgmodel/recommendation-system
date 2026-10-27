"""Controlled query-generation and adaptation comparisons."""
from __future__ import annotations
import csv
from pathlib import Path
import random
from .data import validate_questions
from .evaluation import evaluate_queries, mean, ranking_metrics
from .io import config, new_run, provenance, read_jsonl, resolve, write_json, write_jsonl
from .pipeline import Runtime
from .recommendation import rank

def grouped_intervals(records: list[dict], samples: int, seed: int) -> dict:
    if samples <= 0:
        raise ValueError('Bootstrap sample count must be positive')
    groups = {}
    for row in records:
        groups.setdefault(row['family_id'], []).append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    metrics = ['top1', 'hit3', 'hit5', 'mrr', 'map_at_k', 'ndcg_at_k']
    values = {metric: [] for metric in metrics}
    for _ in range(samples):
        selected = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        for metric in metrics:
            values[metric].append(mean((r[metric] for r in selected)))
    result = {}
    for metric, points in values.items():
        points.sort()
        result[metric] = {'low': points[int(0.025 * (samples - 1))], 'high': points[int(0.975 * (samples - 1))]}
    return {'resampling_unit': 'question_family', 'groups': len(groups), 'samples': samples, 'confidence_level': 0.95, 'intervals': result}

def run_experiment(path, *, generator_override=None, checkpoint_override=None, output_override=None):
    cfg = config(path)
    runtime = Runtime.from_file(resolve(cfg, cfg['runtime']))
    data_path = resolve(cfg, cfg['test'])
    rows = read_jsonl(data_path)
    validate_questions(rows)
    if any(('relevance' not in row for row in rows)):
        raise ValueError('Ranking experiments require independent reference relevance grades')
    choice = generator_override or cfg['generator']
    names = ['template', 'mt5'] if choice == 'both' else [choice]
    generators = {name: runtime.generator(name, checkpoint_override or resolve(cfg, cfg['checkpoint'])) for name in names}
    out = new_run(output_override or resolve(cfg, cfg['output']))
    comparisons = {}
    query_metrics = {}
    all_records = []
    for name, generator in generators.items():
        retrievals = [runtime.retrieve(row['question'], row['context'], generator) for row in rows]
        predictions = [r['query'] or '' for r in retrievals]
        query_metrics[name] = evaluate_queries(rows, predictions, runtime.runner)
        write_jsonl(out / f'{name}_queries.jsonl', [{'id': row['id'], **result} for row, result in zip(rows, retrievals)])
        for adaptive in (False, True):
            arm = name + ('_adaptive' if adaptive else '_static')
            records = []
            for row, retrieval in zip(rows, retrievals):
                if retrieval['status'] == 'executed':
                    pool = [runtime.by_id[key] for key in retrieval['resource_ids']]
                    result = rank(pool, row['context'], retrieval['target_skills'], row.get('learner_state', {}), runtime.config['weights'], adaptive=adaptive, top_k=runtime.config['top_k'], initial_mastery=runtime.config['bkt']['initial'])
                    ids = [r['resource_id'] for r in result['ranked']]
                else:
                    ids = []
                metrics = ranking_metrics(ids, row['relevance'], runtime.config['top_k'])
                records.append({'id': row['id'], 'family_id': row['family_id'], 'arm': arm, 'query_status': retrieval['status'], 'ranked_ids': ids, **metrics})
            metric_names = ['top1', 'hit3', 'hit5', 'mrr', 'map_at_k', 'ndcg_at_k', 'precision_at_k']
            comparisons[arm] = {'n': len(records), 'metrics': {m: mean((r[m] for r in records)) for m in metric_names}, 'nonempty_reference_cases': sum((r['reference_nonempty'] for r in records)), 'bootstrap': grouped_intervals(records, cfg['bootstrap_samples'], cfg['seed'])}
            write_jsonl(out / f'{arm}_ranking.jsonl', records)
            all_records += records
    files = {'test': data_path, 'ontology': runtime.knowledge.ontology, 'catalog': runtime.knowledge.catalog,
             'resources': resolve(runtime.config,runtime.config['resources']),
             'aliases': resolve(runtime.config,runtime.config['aliases']),
             'runtime_config': resolve(cfg,cfg['runtime']), 'experiment_config': path}
    report = provenance(seed=cfg['seed'], files=files, experiment='query_and_adaptation', data_status=sorted({r['data_status'] for r in rows}), reference_kinds=sorted({r.get('reference_kind', 'unspecified') for r in rows}), generators=names, query_metrics=query_metrics, comparisons=comparisons, interpretation='retrieval_and_ranking_only; no educational treatment effect is estimated')
    report['parameters'] = {'weights':runtime.config['weights'],'top_k':runtime.config['top_k'],
                            'initial_mastery':runtime.config['bkt']['initial']}
    write_json(out / 'results.json', report)
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'n', *metric_names])
        writer.writeheader()
        for arm, value in comparisons.items():
            writer.writerow({'arm': arm, 'n': value['n'], **value['metrics']})
    return report
