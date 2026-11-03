"""Production diagnostics around the verified V10 training entry."""
from pathlib import Path
import sys

ROOT = Path('/mnt/workspace/nix3sgh/CVCI2_Res')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]


def main():
    import torch
    import world_models.matched_prior_monitor as monitor
    original_keys, original_gradient = monitor.monitoring_keys, monitor.gradient_monitor
    scopes = {
        'geobev': lambda n: n.startswith('base.geobev.'),
        'bev_bridge': lambda n: n.startswith(('base.bridge.', 'base.occupancy_projection.')),
        'current_readout': lambda n: n.startswith('base.planner.') and
            any(s in n for s in ('.current_query.', '.current_point_logits')),
        'future_distance': lambda n: n.endswith('.future_distance_logit'),
        'world_backbone': lambda n: n.startswith('world.model.img_backbone.'),
        'world_neck': lambda n: n.startswith('world.model.img_neck.'),
        'world_bev_encoder': lambda n: n.startswith('world.model.pts_bbox_head.'),
        'world_future_decoder': lambda n: n.startswith('world.model.future_pred_head.transformer.'),
    }
    extras = ['geobev/occupancy_valid_cells', 'geobev/depth_valid_points',
              'gradient/world_tuned_norm'] + ['gradient/' + key + '_norm' for key in scopes]
    monitor.monitoring_keys = lambda: sorted(set(original_keys() + extras))

    def gradient(model):
        result = original_gradient(model)
        device = next(model.parameters()).device
        for key, selected in scopes.items():
            values = [p.grad.detach().float().norm() for name, p in model.named_parameters()
                      if selected(name) and p.grad is not None]
            norm = torch.stack(values).norm() if values else torch.zeros((), device=device)
            result['gradient/' + key + '_norm'] = torch.stack((norm, torch.ones_like(norm)))
        result['gradient/perception_norm'] = result['gradient/geobev_norm']
        return result

    monitor.gradient_monitor = gradient
    from tools.train_geobev_h20 import main as train
    train()


if __name__ == '__main__':
    main()
