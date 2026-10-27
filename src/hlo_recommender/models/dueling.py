"""Optional Dueling Double DQN experiments on recorded instructional transitions."""
from __future__ import annotations
import math
import random
from ..io import config, new_run, provenance, read_jsonl, resolve, write_json

def validate_transitions(rows, state_dim, action_count):
    if not rows:
        raise ValueError('Transition dataset is empty')
    seen = set()
    for row in rows:
        for key in ('episode_id', 'step', 'state', 'action', 'reward', 'next_state', 'done', 'allowed_actions', 'next_allowed_actions', 'data_status'):
            if key not in row:
                raise ValueError('Missing transition field: ' + key)
        identity = (row['episode_id'], row['step'])
        if identity in seen:
            raise ValueError('Duplicate episode step')
        seen.add(identity)
        for key in ('state', 'next_state'):
            if len(row[key]) != state_dim or any((not math.isfinite(x) for x in row[key])):
                raise ValueError('Invalid policy state')
        for key in ('allowed_actions', 'next_allowed_actions'):
            if len(row[key]) != action_count or any((type(x) is not bool for x in row[key])):
                raise ValueError('Invalid action mask')
        if type(row['action']) is not int or not 0 <= row['action'] < action_count:
            raise ValueError('Invalid action')
        if not row['allowed_actions'][row['action']]:
            raise ValueError('Observed action is not eligible')
        if type(row['done']) is not bool or not math.isfinite(row['reward']):
            raise ValueError('Invalid terminal flag or reward')
        if not row['done'] and (not any(row['next_allowed_actions'])):
            raise ValueError('Nonterminal next state has no eligible action')

def make_network(torch, state_dim, action_count, hidden=64):

    class DuelingNetwork(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(state_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
            self.value = torch.nn.Linear(hidden, 1)
            self.advantage = torch.nn.Linear(hidden, action_count)

        def forward(self, state):
            encoded = self.encoder(state)
            advantage = self.advantage(encoded)
            return self.value(encoded) + advantage - advantage.mean(dim=-1, keepdim=True)
    return DuelingNetwork()

def train_policy(path, *, allow_synthetic=False):
    cfg = config(path)
    for key in ('state_dim', 'action_count', 'hidden', 'steps', 'batch_size', 'target_update', 'report_every'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if not 0 <= cfg['discount'] <= 1 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid policy discount or learning rate')
    train_path, val_path = (resolve(cfg, cfg['train']), resolve(cfg, cfg['validation']))
    train, val = (read_jsonl(train_path), read_jsonl(val_path))
    for rows in (train, val):
        validate_transitions(rows, cfg['state_dim'], cfg['action_count'])
        if not allow_synthetic and any((r['data_status'] != 'reviewed' for r in rows)):
            raise ValueError('Reviewed transitions are required')
    if {r['episode_id'] for r in train} & {r['episode_id'] for r in val}:
        raise ValueError('Episode leakage between train and validation')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install model dependencies with: pip install -e ".[model]"') from exc
    out = new_run(resolve(cfg, cfg['output']))
    torch.manual_seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    online = make_network(torch, cfg['state_dim'], cfg['action_count'], cfg['hidden'])
    target = make_network(torch, cfg['state_dim'], cfg['action_count'], cfg['hidden'])
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=cfg['learning_rate'])

    def tensors(rows):
        return {key: torch.tensor([r[key] for r in rows], dtype=dtype) for key, dtype in {'state': torch.float32, 'next_state': torch.float32, 'action': torch.int64, 'reward': torch.float32, 'done': torch.bool, 'next_allowed_actions': torch.bool}.items()}

    def loss_for(batch, grad=True):
        x = tensors(batch)
        values = online(x['state']).gather(1, x['action'].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = online(x['next_state']).masked_fill(~x['next_allowed_actions'], float('-inf'))
            actions = next_values.argmax(dim=1)
            bootstrap = target(x['next_state']).gather(1, actions.unsqueeze(1)).squeeze(1)
            expected = x['reward'] + cfg['discount'] * (~x['done']).float() * bootstrap
        return torch.nn.functional.smooth_l1_loss(values, expected)
    history = []
    for step in range(cfg['steps']):
        batch = rng.sample(train, min(cfg['batch_size'], len(train)))
        online.train()
        loss = loss_for(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % cfg['target_update'] == 0:
            target.load_state_dict(online.state_dict())
        if (step + 1) % cfg['report_every'] == 0 or step + 1 == cfg['steps']:
            online.eval()
            with torch.no_grad():
                validation_loss = float(loss_for(val))
            history.append({'step': step + 1, 'train_td_loss': float(loss.detach()), 'validation_td_loss': validation_loss})
    torch.save(online.state_dict(), out / 'weights.pt')
    record = provenance(seed=cfg['seed'], files={'train': train_path, 'validation': val_path}, task='instructional_policy', architecture='dueling_double_dqn', status='trained', state_dim=cfg['state_dim'], action_count=cfg['action_count'], hidden=cfg['hidden'], validation_scope='heldout_transition_td_error', history=history, data_status=sorted({r['data_status'] for r in train + val}))
    write_json(out / 'manifest.json', record)
    return record

class DuelingPolicy:

    def __init__(self, directory):
        import json
        from pathlib import Path
        import torch
        path = Path(directory)
        self.torch = torch
        self.manifest = json.loads((path / 'manifest.json').read_text())
        m = self.manifest
        self.model = make_network(torch, m['state_dim'], m['action_count'], m['hidden'])
        self.model.load_state_dict(torch.load(path / 'weights.pt', map_location='cpu', weights_only=True))
        self.model.eval()

    def select(self, state, allowed_actions):
        torch = self.torch
        if len(state) != self.manifest['state_dim'] or len(allowed_actions) != self.manifest['action_count']:
            raise ValueError('Policy input shape mismatch')
        if not any(allowed_actions):
            raise ValueError('No eligible instructional action')
        with torch.inference_mode():
            q = self.model(torch.tensor([state], dtype=torch.float32))[0]
            q = q.masked_fill(~torch.tensor(allowed_actions, dtype=torch.bool), float('-inf'))
        return int(q.argmax())
