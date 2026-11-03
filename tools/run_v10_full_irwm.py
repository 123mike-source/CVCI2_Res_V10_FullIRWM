"""Portable ten-epoch launcher for the exported, live full-IRWM recipe."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(str(temporary), str(path))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def commands(args, recipe, epoch):
    work = args.output / 'run'
    files, rates = recipe['checkpoint_files'], recipe['peak_learning_rates']
    train = [str(args.python), '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=4', str(ROOT / 'tools/train_v10_full_irwm_worker.py'),
        '--work-dir', str(work), '--data-root', str(args.data_root),
        '--base-checkpoint', str(args.models_dir / files['cvci2']),
        '--world-checkpoint', str(args.models_dir / files['irwm']),
        '--world-root', str(args.world_root),
        '--geobev-checkpoint', str(args.models_dir / files['geobev']),
        '--candidate-mode', 'full', '--fresh-res', '--world-tuning-scope', 'full',
        '--depth-loss-weight', str(recipe['loss_weights']['depth']),
        '--batch-size', str(recipe['batch_per_gpu']), '--grad-accum', str(recipe['gradient_accumulation']),
        '--workers', str(recipe['workers_per_gpu']), '--epochs', str(recipe['epochs']),
        '--precision', recipe['precision'], '--optimizer-profile', recipe['optimizer_profile'],
        '--lr', '1e-4', '--image-lr', str(rates['image_backbone']),
        '--pretrained-lr', str(rates['pretrained_other']), '--planner-lr', str(rates['pretrained_planner']),
        '--future-lr', str(rates['new_future']), '--offset-lr', str(rates['new_offset']),
        '--world-lr', str(rates['world_tuned']), '--world-readout-steps', str(recipe['world_readout_steps']),
        '--loss-diagnostics', '--weight-decay', str(recipe['weight_decay']),
        '--ddim-steps', str(recipe['ddim_steps']), '--points-per-map']
    train += [str(v) for v in recipe['points_per_map']]
    train += ['--window-m'] + [str(v) for v in recipe['window_m']]
    train += ['--offset-mode', recipe['offset_mode'], '--future-dropout', str(recipe['future_dropout']),
        '--monitor-every', str(recipe['monitor_every_microsteps']),
        '--log-every', str(recipe['log_every_microsteps']), '--verify-ddp', '--seed', str(recipe['seed']),
        '--warmup-steps', str(recipe['warmup_updates']), '--stop-after-epoch', str(epoch)]
    if epoch > 1:
        train += ['--resume', str(work / 'last.pth')]
    evaluation = [str(args.python), '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=4', str(ROOT / 'tools/eval_geobev_full_coverage.py'),
        '--adapter-checkpoint', str(work / ('epoch_%02d.pth' % epoch)),
        '--output-dir', str(args.output / ('validation/epoch_%02d' % epoch)),
        '--base-checkpoint', str(args.models_dir / files['cvci2']),
        '--world-checkpoint', str(args.models_dir / files['irwm']),
        '--data-root', str(args.data_root), '--world-root', str(args.world_root),
        '--seed', str(recipe['seed']), '--ddim-steps', str(recipe['ddim_steps']),
        '--workers', '2', '--prefetch-factor', '2', '--log-seconds', '30']
    return train, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=ROOT / 'ENV/bin/python')
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/workspace/dataSet/nuscene'))
    parser.add_argument('--models-dir', type=Path, default=ROOT.parent / 'Models')
    parser.add_argument('--world-root', type=Path, default=ROOT / 'world_configs')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--dry-run', action='store_true', help='Print first/last epoch commands without starting jobs or writing files')
    args = parser.parse_args()
    for field in ('python', 'data_root', 'models_dir', 'world_root', 'output'):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    recipe = read(ROOT / 'configs/v10_full_irwm/train.json')
    recorded = read(ROOT / 'configs/v10_full_irwm/recorded_config.json')
    for name, expected in recorded['code_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError('Exported live source differs: ' + name)
    if args.dry_run:
        for epoch in (1, recipe['epochs']):
            train, evaluation = commands(args, recipe, epoch)
            print(json.dumps(dict(epoch=epoch, train=train, evaluation=evaluation), indent=2))
        return
    if args.output.exists():
        raise RuntimeError('Choose a new output directory: ' + str(args.output))
    if not args.python.is_file() or not args.data_root.is_dir() or not args.world_root.is_dir():
        raise RuntimeError('Check --python, --data-root and --world-root')
    for name in recipe['checkpoint_files'].values():
        file = args.models_dir / name
        if not file.is_file():
            raise RuntimeError('Missing original pretrained checkpoint: ' + str(file))
        expected = read(ROOT / 'configs/v10_full_irwm/recorded_recipe.json')['initialization_checkpoints'][name]['sha256']
        if sha256(file) != expected:
            raise RuntimeError('Pretrained checkpoint SHA256 differs: ' + str(file))
    (ROOT / 'ENV').mkdir(exist_ok=True)
    lock = (ROOT / 'ENV/h20_training_launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPUs are occupied; no training was started')
    args.output.mkdir(parents=True)
    write(args.output / 'recipe.json', recipe)
    env = dict(os.environ, **recipe['environment'])
    env.pop('CVCI_DETERMINISTIC_CHECKS', None)
    results = []

    def run(phase, command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(command=command)) + '\n')
            stream.flush()
            child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
            write(args.output / 'state.json', dict(phase=phase, pid=os.getpid(), active_pid=child.pid,
                updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat()))
            if child.wait():
                raise RuntimeError(phase + ' failed; see ' + str(log))

    try:
        for epoch in range(1, recipe['epochs'] + 1):
            train, evaluation = commands(args, recipe, epoch)
            run('training_epoch%02d' % epoch, train, args.output / 'run/train.log')
            actual = read(args.output / 'run/config.json')
            groups = {g['name']: g['peak_lr'] for g in actual['training_optimization']['optimizer_groups']}
            assert groups == recipe['peak_learning_rates'], groups
            assert actual['config']['fresh_res'] and not actual['config']['warm_start_geobev']
            assert not actual['config']['warm_start']
            assert actual['world_tuning_config']['parameter_scope'] == 'all_registered_irwm_parameters'
            assert actual['nominal_effective_batch'] == recipe['effective_batch']
            assert actual['fresh_initialization_audit']['prior_res_checkpoint_loaded'] is False
            assert actual['lr_schedule']['total_updates'] == recipe['recorded_total_updates']
            run('validation_epoch%02d' % epoch, evaluation, args.output / ('validation/epoch_%02d.log' % epoch))
            metrics = read(args.output / ('validation/epoch_%02d/metrics.json' % epoch))
            assert metrics['sample_coverage_check'] == 'PASS' and not metrics['partial']
            for key in ('seen_frames', 'scored_frames', 'rolling_frames', 'discarded_incomplete_future_frames'):
                assert metrics[key] == recipe['validation'][key], key
            value = dict(epoch=epoch, checkpoint=str(args.output / ('run/epoch_%02d.pth' % epoch)),
                rolling_4585_L2_m=metrics['retained_original_subset']['avg_L2_m'],
                full_valid_5119_L2_m=metrics['avg_L2_m'])
            assert math.isfinite(value['rolling_4585_L2_m']) and math.isfinite(value['full_valid_5119_L2_m'])
            results.append(value)
            write(args.output / 'epoch_metrics.json', results)
            write(args.output / 'best.json', min(results, key=lambda v: v['rolling_4585_L2_m']))
        write(args.output / 'state.json', dict(phase='complete', completed_epochs=recipe['epochs']))
    except Exception:
        write(args.output / 'state.json', dict(phase='failed', error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
