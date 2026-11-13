"""Small dual TCN with rank-specific regression heads; CPU reproducible training."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalLayer(nn.Module):
    def __init__(self, inputs, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(inputs, width, 3, padding=2*dilation, dilation=dilation)
        self.skip = nn.Conv1d(inputs, width, 1) if inputs != width else nn.Identity()
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        return self.skip(x) + self.dropout(F.gelu(self.conv(x)[..., :x.shape[-1]]))


class DualTCN(nn.Module):
    def __init__(self, tab_features, width=24):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(*[
            CausalLayer(2 if i == 0 else width, width, 2**i) for i in range(3)]) for _ in range(2)])
        self.tab = nn.Sequential(nn.Linear(tab_features, 32), nn.GELU(), nn.LayerNorm(32), nn.Dropout(.1))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(4*width+32, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(2)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, sequence, tab, rank):
        encoded = [encoder(sequence[:, i].transpose(1, 2)) for i, encoder in enumerate(self.encoders)]
        # The last state covers local dynamics; pooling exposes the entire prefix.
        dynamic = [torch.cat([value[:, :, -1], value.mean(dim=-1)], dim=1) for value in encoded]
        state = torch.cat(dynamic+[self.tab(tab)], dim=1)
        output = torch.cat([head(state) for head in self.heads], dim=1)
        return output.gather(1, rank[:, None]).squeeze(1)


def infer(model, values, batch_size=256):
    model.eval()
    with torch.no_grad():
        values = {k: torch.as_tensor(v) for k, v in values.items()}
        result = np.concatenate([model(values['sequence'][i:i+batch_size], values['tab'][i:i+batch_size],
                                       values['rank'][i:i+batch_size]).numpy()
                                 for i in range(0, len(values['rank']), batch_size)])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite TCN predictions')
    return result


def train_seed(train, valid, target, weights, validation_score, *, seed, max_epochs, checkpoint):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = DualTCN(train['tab'].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.005)
    values = {k: torch.as_tensor(v) for k, v in train.items()}
    y = torch.tensor(target, dtype=torch.float32); w = torch.tensor(weights, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, trace = float('inf'), 0, None, []
    for epoch in range(1, max_epochs+1):
        model.train(); order = torch.randperm(len(y), generator=rng); loss_sum = 0.
        for begin in range(0, len(order), 128):
            idx = order[begin:begin+128]
            optimizer.zero_grad(set_to_none=True)
            pred = model(values['sequence'][idx], values['tab'][idx], values['rank'][idx])
            loss = ((pred-y[idx]).square()*w[idx]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            loss_sum += loss.item()*len(idx)
        score = validation_score(infer(model, valid))
        trace.append({'epoch': epoch, 'train_weighted_mse': loss_sum/len(y), 'validation_event_mape': score})
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'TCN seed={seed} epoch={epoch} validation={score:.4f} best={best:.4f} best_epoch={best_epoch}', flush=True)
        if epoch >= 30 and epoch-best_epoch >= 20:
            break
    torch.save({'state': best_state, 'tab_features': train['tab'].shape[1], 'width': 24, 'seed': seed}, checkpoint)
    reloaded = load_model(checkpoint)
    model.load_state_dict(best_state)
    np.testing.assert_allclose(infer(model, valid), infer(reloaded, valid), rtol=1e-7, atol=1e-7)
    return reloaded, {'seed': seed, 'best_epoch': best_epoch, 'epochs_run': epoch,
        'validation_event_mape': best, 'parameters': sum(p.numel() for p in model.parameters()), 'trace': trace}


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = DualTCN(saved['tab_features'], saved['width'])
    model.load_state_dict(saved['state'])
    model.eval()
    return model
