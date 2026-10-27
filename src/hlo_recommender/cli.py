"""Research workflow command line interface."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .data import validate_events, validate_partitions, validate_questions
from .development import create_development_data
from .evaluation import binary_metrics, evaluate_queries, mean, ranking_metrics
from .experiments import run_experiment
from .io import config, grouped_split, new_run, read_jsonl, resolve, write_json, write_jsonl, provenance
from .knowledge import HLO
from .models.bkt import BKTParameters, LearnerStore, load_parameters, predict_events
from .models.mt5 import download_model
from .pipeline import Runtime
from .training import train_bkt, train_mt5

def parser():
    p = argparse.ArgumentParser(prog='hlo', description='HLO recommendation research workflows')
    p.add_argument('--config', default='configs/default.toml', help='Runtime configuration')
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('inspect')
    data = commands.add_parser('data').add_subparsers(dest='action', required=True)
    development = data.add_parser('development')
    development.add_argument('--seed', type=int, default=42)
    check = data.add_parser('validate')
    check.add_argument('--directory', required=True)
    split = data.add_parser('split')
    split.add_argument('--input', required=True)
    split.add_argument('--kind', choices=['questions', 'assessments'], required=True)
    split.add_argument('--output', required=True)
    split.add_argument('--seed', type=int, default=42)
    download = commands.add_parser('download')
    download.add_argument('--model', default='google/mt5-small')
    download.add_argument('--revision', default='main')
    download.add_argument('--output', required=True)
    train = commands.add_parser('train').add_subparsers(dest='model', required=True)
    for name in ('mt5', 'bkt', 'policy'):
        sub = train.add_parser(name)
        sub.add_argument('--config', dest='training_config', required=True)
        sub.add_argument('--allow-synthetic', action='store_true')
        if name == 'mt5':
            sub.add_argument('--check-only', action='store_true')
    evaluate = commands.add_parser('evaluate').add_subparsers(dest='target', required=True)
    q = evaluate.add_parser('queries')
    q.add_argument('--data', required=True)
    q.add_argument('--predictions')
    q.add_argument('--generator', choices=['template', 'mt5'], default='mt5')
    q.add_argument('--checkpoint')
    q.add_argument('--output', required=True)
    b = evaluate.add_parser('bkt')
    b.add_argument('--data', required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--output', required=True)
    r = evaluate.add_parser('ranking')
    r.add_argument('--data', required=True)
    r.add_argument('--predictions', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--top-k', type=int, default=5)
    experiment = commands.add_parser('experiment')
    experiment.add_argument('--config', dest='experiment_config', required=True)
    experiment.add_argument('--generator', choices=['template', 'mt5', 'both'])
    experiment.add_argument('--checkpoint')
    experiment.add_argument('--output')
    rec = commands.add_parser('recommend')
    rec.add_argument('--question', required=True)
    rec.add_argument('--level', choices=['Junior', 'Senior', 'Fellow'], required=True)
    rec.add_argument('--minutes', type=int, required=True)
    rec.add_argument('--generator', choices=['mt5', 'template'], default='mt5')
    rec.add_argument('--checkpoint')
    rec.add_argument('--learner', default='learner-001')
    rec.add_argument('--database')
    rec.add_argument('--bkt-model')
    rec.add_argument('--static', action='store_true')
    rec.add_argument('--output')
    observe = commands.add_parser('observe')
    observe.add_argument('--events', required=True)
    observe.add_argument('--database', required=True)
    observe.add_argument('--bkt-model')
    demo = commands.add_parser('demo')
    demo.add_argument('--output', required=True)
    return p

def aligned_predictions(rows, path, key):
    predictions = read_jsonl(path)
    mapping = {r['id']: r[key] for r in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r['id'] for r in rows}:
        raise ValueError('Prediction IDs must match dataset IDs exactly once')
    return [mapping[row['id']] for row in rows]

def run(args):
    if args.command == 'download':
        return download_model(args.model, args.output, args.revision)
    if args.command == 'train':
        if args.model == 'mt5':
            return train_mt5(args.training_config, allow_synthetic=args.allow_synthetic, check_only=args.check_only)
        if args.model == 'bkt':
            return train_bkt(args.training_config, allow_synthetic=args.allow_synthetic)
        from .models.dueling import train_policy
        return train_policy(args.training_config, allow_synthetic=args.allow_synthetic)
    if args.command == 'experiment':
        return run_experiment(args.experiment_config, generator_override=args.generator, checkpoint_override=args.checkpoint, output_override=args.output)
    if args.command == 'data':
        root = Path(args.config).resolve().parent.parent
        if args.action == 'development':
            return create_development_data(root, args.seed)
        if args.action == 'split':
            rows = read_jsonl(args.input)
            if args.kind == 'questions':
                validate_questions(rows)
                key = 'family_id'
            else:
                validate_events(rows)
                key = 'learner_id'
            output = new_run(args.output)
            split = grouped_split(rows, key, args.seed)
            for name, items in split.items():
                write_jsonl(output / f'{args.kind}.{name}.jsonl', items)
            return {name: len(items) for name, items in split.items()}
        directory = Path(args.directory)
        partitions = {name: read_jsonl(directory / f'questions.{name}.jsonl') for name in ('train', 'validation', 'test')}
        result = validate_partitions(partitions)
        event_files = [directory / f'assessments.{name}.jsonl' for name in ('train', 'validation', 'test')]
        if all((path.exists() for path in event_files)):
            seen = set()
            for path in event_files:
                events = read_jsonl(path)
                validate_events(events)
                learners = {r['learner_id'] for r in events}
                if learners & seen:
                    raise ValueError('Learner leakage in assessment partitions')
                seen |= learners
        return {'status': 'passed', 'questions': result}
    if args.command == 'evaluate' and args.target == 'bkt':
        rows = read_jsonl(args.data)
        predictions, metrics = predict_events(rows, load_parameters(args.model))
        metadata = json.loads(Path(args.model).read_text())
        if 'constant_probability' in metadata:
            metrics['constant_baseline'] = binary_metrics([r['correct'] for r in predictions], [metadata['constant_probability']] * len(predictions))
        metrics.update(provenance(seed=metadata.get('seed',42),files={'assessments':args.data,'model':args.model},
                                  task='knowledge_tracing_evaluation',
                                  data_status=sorted({row['data_status'] for row in rows})))
        output = new_run(args.output)
        write_json(output / 'metrics.json', metrics)
        write_jsonl(output / 'predictions.jsonl', predictions)
        return metrics
    if args.command == 'evaluate' and args.target == 'ranking':
        rows = read_jsonl(args.data)
        if not rows:
            raise ValueError('Ranking dataset is empty')
        predictions = aligned_predictions(rows, args.predictions, 'ranked_ids')
        records = [{'id': r['id'], **ranking_metrics(ids, r['relevance'], args.top_k)} for r, ids in zip(rows, predictions)]
        names = ['top1', 'hit3', 'hit5', 'mrr', 'map_at_k', 'ndcg_at_k', 'precision_at_k']
        metrics = {'n': len(records), 'metrics': {k: mean((r[k] for r in records)) for k in names}}
        metrics.update(provenance(seed=42,files={'data':args.data,'predictions':args.predictions},
                                  task='ranking_evaluation',top_k=args.top_k))
        output = new_run(args.output)
        write_json(output / 'metrics.json', metrics)
        write_jsonl(output / 'records.jsonl', records)
        return metrics
    cfg = config(args.config)
    runtime = Runtime(cfg)
    if args.command == 'inspect':
        return runtime.knowledge.statistics()
    if args.command == 'evaluate':
        rows = read_jsonl(args.data)
        validate_questions(rows)
        if args.predictions:
            predictions = aligned_predictions(rows, args.predictions, 'sparql')
        else:
            generator = runtime.generator(args.generator, args.checkpoint)
            predictions = [runtime.retrieve(r['question'], r['context'], generator)['query'] or '' for r in rows]
        result = evaluate_queries(rows, predictions, runtime.runner)
        result['data_status'] = sorted({r['data_status'] for r in rows})
        files={'data':args.data,'ontology':runtime.knowledge.ontology,'catalog':runtime.knowledge.catalog,
               'runtime_config':args.config}
        if args.predictions:files['predictions']=args.predictions
        elif args.checkpoint:files['checkpoint_manifest']=Path(args.checkpoint)/'manifest.json'
        result.update(provenance(seed=cfg['seed'],files=files,task='query_evaluation'))
        write_json(Path(new_run(args.output)) / 'metrics.json', result)
        return result
    if args.command == 'observe':
        store = LearnerStore(args.database, load_parameters(args.bkt_model) if args.bkt_model else BKTParameters(**cfg['bkt']))
        try:
            result = [store.observe(row) for row in read_jsonl(args.events)]
        finally:
            store.close()
        return {'events': len(result), 'updates': result}
    if args.command == 'demo':
        return demo(runtime, args.output)
    params = load_parameters(args.bkt_model) if args.bkt_model else BKTParameters(**cfg['bkt'])
    store = LearnerStore(args.database or resolve(cfg, cfg['learner_database']), params)
    try:
        state = store.snapshot(args.learner)
    finally:
        store.close()
    result = runtime.recommend(args.question, {'level': args.level, 'minutes': args.minutes}, runtime.generator(args.generator, args.checkpoint), state=state, adaptive=not args.static)
    if args.output:
        write_json(args.output, result)
    return result

def demo(runtime, output):
    from datetime import datetime, timedelta, timezone
    out = new_run(output)
    store = LearnerStore(out / 'learner.sqlite', BKTParameters(**runtime.config['bkt']))
    question = 'Find resources for arterial blood gas interpretation.'
    context = {'level': 'Junior', 'minutes': 15}
    skill = str(HLO.K002)
    generator = runtime.generator('template')
    events = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    try:
        for i in range(2):
            event = {'event_id': f'demo-{i}', 'learner_id': 'development-learner', 'skill_id': skill, 'correct': False, 'timestamp': (start + timedelta(minutes=i)).isoformat(), 'data_status': 'synthetic'}
            store.observe(event)
            events.append(event)
        before = runtime.recommend(question, context, generator, state=store.snapshot('development-learner'))
        for i in range(2, 10):
            event = {'event_id': f'demo-{i}', 'learner_id': 'development-learner', 'skill_id': skill, 'correct': True, 'timestamp': (start + timedelta(minutes=i)).isoformat(), 'data_status': 'synthetic'}
            store.observe(event)
            events.append(event)
        after = runtime.recommend(question, context, generator, state=store.snapshot('development-learner'))
    finally:
        store.close()
    report = {'mode': 'synthetic_integration', 'query_generator': 'template_baseline', 'bkt_parameters': 'configured_initial_parameters', 'question': question, 'context': context, 'before': before, 'after': after}
    write_json(out / 'recommendations.json', report)
    write_jsonl(out / 'assessments.jsonl', events)
    return report

def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
