"""Explicit optimizer groups and opt-in training diagnostics for v6 weights."""
import math
import numpy as np
import torch

LOSS_DIAGNOSTIC_KEYS = tuple('layer_%02d_x0' % i for i in range(1, 13)) + (
    'all_layers_t00_24', 'all_layers_t25_49', 'all_layers_t50_74',
    'all_layers_t75_99', 'all_layers_t99', 'last_layer_t99')


def optimizer_groups(model, profile='legacy', lr=1e-4, image_lr=None,
                     pretrained_lr=None, planner_lr=None, future_lr=None, offset_lr=None, world_lr=None):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or profile not in ('legacy', 'differential'):
        raise ValueError('invalid optimizer profile/trainable scope')
    if profile == 'legacy':
        if any(x is not None for x in (image_lr, pretrained_lr, planner_lr, future_lr, offset_lr)):
            raise ValueError('explicit group LRs require differential profile')
        rates = {'image_backbone': lr*.5, 'legacy_other': lr}
    else:
        new_rate = lr if future_lr is None else future_lr
        rates = {'image_backbone': 5e-6 if image_lr is None else image_lr,
                 'pretrained_other': 1e-5 if pretrained_lr is None else pretrained_lr,
                 'pretrained_planner': 2e-5 if planner_lr is None else planner_lr,
                 'new_future': new_rate, 'new_offset': new_rate if offset_lr is None else offset_lr}
    if hasattr(model, 'geobev_config'):
        if profile != 'differential': raise ValueError('GeoBEV requires differential learning rates')
        rates['new_bev_bridge'] = lr if future_lr is None else future_lr
        rates['geobev_depth_heads'] = rates['pretrained_other']
    if any(n.startswith('world.') for n, p in named):
        if profile != 'differential' or getattr(model, 'world_tuning_config', None) is None:
            raise AssertionError('World gradients need an explicit differential tuning contract')
        rates['world_tuned'] = 1e-6 if world_lr is None else world_lr
    if getattr(model, 'candidate_config', None):
        rates['new_current_readout'] = lr if future_lr is None else future_lr
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError('learning rates must be positive finite values')
    groups = {name: [] for name in rates}
    names = {name: [] for name in rates}
    for name, parameter in named:
        if name.startswith('world.'):
            if not model.is_trainable(name): raise AssertionError('Unapproved world parameter: '+name)
            group = 'world_tuned'
        elif profile == 'legacy':
            group = 'image_backbone' if name.startswith('base.backbone.image.') else 'legacy_other'
        elif name.startswith('future_encoder.offset.'):
            group = 'new_offset'
        elif name.startswith('future_encoder.') or (name.startswith('base.planner.') and '.future_' in name):
            group = 'new_future'
        elif name.startswith(('base.geobev.img_view_transformer.depth_net.depth_head.', 'base.geobev.img_view_transformer.depth_convs.')):
            group = 'geobev_depth_heads'
        elif name.startswith(('base.bridge.', 'base.occupancy_projection.')):
            group = 'new_bev_bridge'
        elif name.startswith(('base.backbone.image.', 'base.geobev.img_backbone.')):
            group = 'image_backbone'
        elif name.startswith('base.planner.') and any(part in name for part in ('.current_query.', '.current_point_logits')):
            group = 'new_current_readout'
        elif name.startswith('base.planner.'):
            group = 'pretrained_planner'
        elif name.startswith('base.'):
            group = 'pretrained_other'
        else:
            raise AssertionError('unclassified trainable tensor: '+name)
        groups[group].append(parameter)
        names[group].append(name)
    result = [{'name': group, 'params': groups[group], 'lr': rates[group]} for group in rates if groups[group]]
    # Hard-filter foreground rows have no gradient; avoid unsupervised AdamW drift.
    for group in result:
        if group['name'] == 'geobev_depth_heads': group['weight_decay'] = 0.
    seen = [id(p) for group in result for p in group['params']]
    assert len(seen) == len(set(seen)) == len(named)
    assert set(seen) == {id(p) for _, p in named}
    manifest = [{'name': group['name'], 'peak_lr': group['lr'],
                 'parameter_count': sum(p.numel() for p in group['params']),
                 'tensor_count': len(group['params']), 'parameter_names': names[group['name']]} for group in result]
    return result, manifest


def configure_training(model, args):
    fraction = float(args.t99_fraction)
    weights = tuple(float(x) for x in args.auxiliary_weights)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('t99 fraction must be in [0,1]')
    if len(weights) != 3 or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError('invalid auxiliary weights')
    if args.world_readout_steps not in (5, 6):
        raise ValueError('world readout steps must be 5 or 6')
    model.base.planner.training_t99_fraction = fraction
    model.base.planner.loss_diagnostics_enabled = bool(args.loss_diagnostics)
    model.training_auxiliary_weights = weights
    if model.world is not None:
        model.world.readout_steps = args.world_readout_steps
        if getattr(model, 'world_tuning_config', None) is not None:
            model.world.checkpoint_future = not getattr(args, 'no_world_checkpoint', False)
    return {'world_activation_checkpointing': bool(getattr(model.world, 'checkpoint_future', False)) if model.world_tuning_config is not None else False,
            'optimizer_profile': args.optimizer_profile, 't99_fraction': fraction,
            'auxiliary_weights': list(weights), 'loss_diagnostics': bool(args.loss_diagnostics),
            'world_readout_steps': args.world_readout_steps,
            'loss_diagnostic_keys': list(LOSS_DIAGNOSTIC_KEYS)}


def restore_fused_optimizer_devices(optimizer):
    """Torch 2.0 restores AdamW step tensors to CPU unless explicitly repaired."""
    moved = 0
    for group in optimizer.param_groups:
        if not group.get('fused', False):
            continue
        for parameter in group['params']:
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.device != parameter.device:
                    optimizer.state[parameter][key] = value.to(device=parameter.device)
                    moved += 1
    return moved


def exact_state_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(exact_state_equal(x,y) for x,y in zip(a,b))
    return a == b


def verify_restored_state(model, optimizer, scheduler, sampler, current_rng, saved, rank):
    from temporal_matched_deform_model import is_trainable_state
    actual = {n:v for n,v in model.state_dict().items() if is_trainable_state(n, model)}
    checks = dict(model=exact_state_equal(actual, saved['trainable']),
        optimizer=exact_state_equal(optimizer.state_dict(), saved['optimizer']),
        scheduler=exact_state_equal(scheduler.state_dict(), saved['scheduler']),
        sampler=exact_state_equal(sampler.state_dict(), saved['rank_states'][rank]['sampler']),
        rolling=exact_state_equal(model.stream.state_dict(), saved['rank_states'][rank]['rolling']),
        rng=exact_state_equal(current_rng, saved['rank_states'][rank]['rng']))
    if not all(checks.values()):
        raise AssertionError('Checkpoint restoration lost state: '+str(checks))
    return dict(rank=rank,status='PASS',checks=checks)


@torch.no_grad()
def loss_diagnostics(predictions, clean, valid, timestep):
    # Detached reductions add no random draws or gradients to the training loss.
    squared = torch.stack([((p.detach().float()-clean).square()*valid).sum((1, 2)) for p in predictions])
    denominator = 2*valid.sum((1, 2))
    pairs = [torch.stack((value.sum(), denominator.sum())) for value in squared]
    average = squared.mean(0)
    for low, high in ((0,24), (25,49), (50,74), (75,99), (99,99)):
        select = ((timestep >= low) & (timestep <= high)).float()
        pairs.append(torch.stack(((average*select).sum(), (denominator*select).sum())))
    select = (timestep == 99).float()
    pairs.append(torch.stack(((squared[-1]*select).sum(), (denominator*select).sum())))
    return torch.stack(pairs)
