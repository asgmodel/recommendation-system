"""Separate training procedures for the query generator and mastery model."""
from __future__ import annotations
from dataclasses import asdict
import json
import math
import random
from .data import validate_events, validate_partitions
from .evaluation import binary_metrics, evaluate_queries
from .io import config, new_run, provenance, read_jsonl, resolve, write_json, write_jsonl
from .models.bkt import fit_bkt, predict_events
from .models.mt5 import decode_candidates, dependencies, serialize_input
from .pipeline import Runtime

def train_bkt(path, *, allow_synthetic=False):
    cfg = config(path)
    train_path, val_path = (resolve(cfg, cfg['train']), resolve(cfg, cfg['validation']))
    train, val = (read_jsonl(train_path), read_jsonl(val_path))
    for rows in (train, val):
        validate_events(rows)
        if not allow_synthetic and any((r['data_status'] != 'reviewed' for r in rows)):
            raise ValueError('Use reviewed assessments, or --allow-synthetic for development')
    if {r['learner_id'] for r in train} & {r['learner_id'] for r in val}:
        raise ValueError('Learner leakage between BKT train and validation')
    if {r['event_id'] for r in train} & {r['event_id'] for r in val}:
        raise ValueError('Assessment event leakage')
    out = new_run(resolve(cfg, cfg['output']))
    params, fit = fit_bkt(train, seed=cfg['seed'], restarts=cfg['restarts'])
    predictions, metrics = predict_events(val, params)
    base = sum((r['correct'] for r in train)) / len(train)
    baseline = binary_metrics([r['correct'] for r in val], [base] * len(val))
    record = provenance(seed=cfg['seed'], files={'train': train_path, 'validation': val_path}, task='knowledge_tracing', status='trained', data_status=sorted({r['data_status'] for r in train + val}), parameters=asdict(params), fit=fit, validation=metrics, validation_constant_baseline=baseline, constant_probability=base)
    write_json(out / 'model.json', record)
    write_jsonl(out / 'validation_predictions.jsonl', predictions)
    return record

def train_mt5(path, *, allow_synthetic=False, check_only=False):
    cfg = config(path)
    for key in ('epochs', 'batch_size', 'max_input_tokens', 'max_target_tokens', 'num_beams', 'patience'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if not 0 < cfg['learning_rate'] or cfg['weight_decay'] < 0 or (not 0 <= cfg['warmup_fraction'] < 1):
        raise ValueError('Invalid optimizer or warmup settings')
    runtime = Runtime.from_file(resolve(cfg, cfg['runtime']))
    train_path, val_path = (resolve(cfg, cfg['train']), resolve(cfg, cfg['validation']))
    train, val = (read_jsonl(train_path), read_jsonl(val_path))
    counts = validate_partitions({'train': train, 'validation': val}, require_reviewed=not allow_synthetic)
    schemas = {row['id']: runtime.grounder.schema_text(row['question']) for row in train + val}
    for row in train + val:
        runtime.runner.execute(row['sparql'])
    if check_only:
        return {'status': 'data_validated', 'counts': counts, 'model_loaded': False}
    torch, tokenizer_cls, model_cls = dependencies()
    base = resolve(cfg, cfg['base_checkpoint'])
    if not base.is_dir():
        raise ValueError('Download the base mT5 checkpoint before training')
    out = new_run(resolve(cfg, cfg['output']))
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = tokenizer_cls.from_pretrained(base, local_files_only=True)
    model = model_cls.from_pretrained(base, local_files_only=True)
    if model.config.model_type != 'mt5':
        raise ValueError('The base checkpoint is not mT5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    steps = math.ceil(len(train) / cfg['batch_size']) * cfg['epochs']
    warmup = max(1, round(steps * cfg['warmup_fraction']))

    def scale(step):
        return step / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    generation = {key: cfg[key] for key in ('max_input_tokens', 'max_target_tokens', 'num_beams')}
    best, stale, history = (-1.0, 0, [])
    for epoch in range(cfg['epochs']):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), cfg['batch_size']):
            batch = [train[i] for i in order[start:start + cfg['batch_size']]]
            inputs = tokenizer([serialize_input(r['question'], schemas[r['id']], r['context']) for r in batch], padding=True, return_tensors='pt')
            targets = tokenizer(text_target=[r['sparql'] for r in batch], padding=True, return_tensors='pt')
            if inputs['input_ids'].shape[-1] > cfg['max_input_tokens']:
                raise ValueError('Training input exceeds token budget')
            if targets['input_ids'].shape[-1] > cfg['max_target_tokens']:
                raise ValueError('Training target exceeds token budget')
            if tokenizer.unk_token_id is not None and (targets['input_ids'] == tokenizer.unk_token_id).any():
                raise ValueError('SPARQL target contains unrepresentable tokenizer symbols')
            labels = targets['input_ids'].clone()
            labels[targets['attention_mask'] == 0] = -100
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{k: v.to(device) for k, v in inputs.items()}, labels=labels.to(device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        predictions = []
        for row in val:
            candidates = decode_candidates(model, tokenizer, serialize_input(row['question'], schemas[row['id']], row['context']), **generation)
            selected = runtime.select(candidates)
            predictions.append(selected['query'] or '')
        metrics = evaluate_queries(val, predictions, runtime.runner)
        score = metrics['nonempty_f1']
        if score is None:
            raise ValueError('Validation requires nonempty reference queries')
        history.append({'epoch': epoch + 1, 'train_loss': sum(losses) / len(losses), 'validation': metrics})
        write_json(out / 'history.json', history)
        if score > best:
            best, stale = (score, 0)
            checkpoint = out / 'checkpoint'
            checkpoint.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            manifest = provenance(seed=cfg['seed'], files={'train': train_path, 'validation': val_path, 'ontology': runtime.knowledge.ontology, 'catalog': runtime.knowledge.catalog}, task='text_to_sparql', status='trained', best_epoch=epoch + 1, ontology_sha256=runtime.knowledge.ontology_sha256, grounder_sha256=runtime.grounder.sha256, generation=generation, data_status=sorted({r['data_status'] for r in train + val}), validation_nonempty_f1=score, training_config=cfg)
            write_json(checkpoint / 'manifest.json', manifest)
            write_jsonl(out / 'best_validation_predictions.jsonl', [{'id': row['id'], 'sparql': q} for row, q in zip(val, predictions)])
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    return {'status': 'trained', 'best_validation_nonempty_f1': best, 'checkpoint': str(out / 'checkpoint')}
