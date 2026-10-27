"""Local mT5 generation with checkpoint and ontology version binding."""
from __future__ import annotations
import json
from pathlib import Path

def serialize_input(question: str, schema: str, context: dict) -> str:
    allowed = {key: context[key] for key in ('level', 'minutes', 'location', 'activity') if key in context}
    return 'QUESTION\n' + question.strip() + '\nSCHEMA\n' + schema + '\nCONTEXT\n' + json.dumps(allowed, sort_keys=True)

def dependencies():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install model dependencies with: pip install -e ".[model]"') from exc
    return (torch, AutoTokenizer, AutoModelForSeq2SeqLM)

def decode_candidates(model, tokenizer, text, *, max_input_tokens=2048, max_target_tokens=384, num_beams=3):
    import torch
    tokens = tokenizer(text, return_tensors='pt')
    if tokens['input_ids'].shape[-1] > max_input_tokens:
        raise ValueError('Input exceeds token budget; narrow the schema fragment')
    device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model.generate(**{k: v.to(device) for k, v in tokens.items()}, do_sample=False, num_beams=num_beams, num_return_sequences=num_beams, max_new_tokens=max_target_tokens)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

class MT5Generator:
    name = 'mt5'

    def __init__(self, checkpoint: str | Path, ontology_sha256: str, grounder_sha256: str):
        checkpoint = Path(checkpoint)
        path = checkpoint / 'manifest.json'
        if not path.is_file():
            raise ValueError('A fine-tuned mT5 checkpoint and manifest.json are required')
        self.manifest = json.loads(path.read_text())
        if self.manifest.get('task') != 'text_to_sparql' or self.manifest.get('status') != 'trained':
            raise ValueError('Checkpoint is not trained for Text-to-SPARQL')
        if self.manifest['ontology_sha256'] != ontology_sha256:
            raise ValueError('Checkpoint ontology version mismatch')
        if self.manifest['grounder_sha256'] != grounder_sha256:
            raise ValueError('Checkpoint alias version mismatch')
        torch, tokenizer, model = dependencies()
        self.tokenizer = tokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = model.from_pretrained(checkpoint, local_files_only=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.settings = self.manifest['generation']

    def generate(self, question, schema, context):
        return decode_candidates(self.model, self.tokenizer, serialize_input(question, schema, context), **self.settings)

def download_model(model_id: str, output: str | Path, revision='main') -> dict:
    from ..io import new_run, write_json
    _, tokenizer_cls, model_cls = dependencies()
    output = new_run(output)
    tokenizer = tokenizer_cls.from_pretrained(model_id, revision=revision)
    model = model_cls.from_pretrained(model_id, revision=revision)
    if model.config.model_type != 'mt5':
        raise ValueError('The query model must use the mT5 architecture')
    tokenizer.save_pretrained(output)
    model.save_pretrained(output, safe_serialization=True)
    manifest = {'model_id': model_id, 'requested_revision': revision, 'resolved_revision': getattr(model.config, '_commit_hash', None), 'task': 'pretrained_base', 'status': 'pretrained'}
    write_json(output / 'base_manifest.json', manifest)
    return manifest
