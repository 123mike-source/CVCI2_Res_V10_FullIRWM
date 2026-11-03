"""V10 full, original pretraining, differential LRs and the full IRWM at 1e-6."""
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path('/mnt/workspace/nix3sgh/CVCI2_Res')
OUT = ROOT / 'experiments/h20_v10_fresh_fullworld10_20261103_01'
IMPL = ROOT / 'experiments/h20_v10_full_irwm_implementation_20261103_01'
PREVIOUS = ROOT / 'experiments/h20_geobev_fresh10_20261101_01'
MODELS = ROOT.parent / 'Models'
PYTHON = str(ROOT / 'ENV/bin/python')
RATES = {'image_backbone': 5e-6, 'pretrained_other': 1e-5,
         'pretrained_planner': 2e-5, 'new_future': 1e-4, 'new_offset': 1e-4,
         'new_bev_bridge': 1e-4, 'geobev_depth_heads': 1e-5,
         'world_tuned': 1e-6, 'new_current_readout': 1e-4}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(str(temporary), str(path))


def main():
    lock = (ROOT / 'ENV/h20_training_launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started, timings = time.monotonic(), []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CVCI_PREFETCH_FACTOR='2',
        PYTHONUNBUFFERED='1', NCCL_DEBUG='WARN', CVCI_RECORD_INPUT_HASHES='1',
        CVCI_WORLD_READOUT_STEPS='5', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.pop('CVCI_DETERMINISTIC_CHECKS', None)

    def status(phase, **details):
        value = dict(phase=phase, pid=os.getpid(), output_root=str(OUT),
            architecture_version=10, candidate_mode='full', world_tuning_scope='full', epochs=10,
            updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            elapsed_seconds=time.monotonic()-started, phase_timings=timings, **details)
        write(OUT / 'state.json', value)
        write(ROOT / 'ENV/h20_v10_latest.json', value)
        print('V10_STATE ' + json.dumps(value, ensure_ascii=False), flush=True)

    def unchanged():
        for name, item in read(OUT / 'approved_source_manifest.json').items():
            assert digest(ROOT / name) == item['after_sha256'], name
        for name, sha in read(OUT / 'runner_manifest.json').items():
            assert digest(OUT / 'scripts' / name) == sha, name

    def run(phase, args, log, append=False):
        unchanged()
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError('GPUs occupied before ' + phase + ': ' + busy)
        log.parent.mkdir(parents=True, exist_ok=True)
        before = time.monotonic()
        with log.open('a' if append else 'x') as stream:
            stream.write(json.dumps(dict(command=args)) + '\n')
            stream.flush()
            child = subprocess.Popen(args, cwd=str(ROOT), env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            status(phase, active_pid=child.pid, log=str(log), command=args)
            code = child.wait()
        timings.append(dict(phase=phase, seconds=time.monotonic()-before, exit_code=code))
        if code:
            raise RuntimeError(phase + ' failed: ' + str(log))

    def command(work, stop_epoch=1, smoke=False):
        args = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            str(OUT / 'scripts/train_v10_production.py'), '--work-dir', str(work),
            '--data-root', '/mnt/workspace/dataSet/nuscene',
            '--base-checkpoint', str(MODELS / 'best_endpoint_1step.pth'),
            '--world-checkpoint', str(MODELS / 'MMO_MSO_with_plan_tightly_coupled.pth'),
            '--world-root', str(ROOT / 'world_configs'),
            '--geobev-checkpoint', str(MODELS / 'geobev-r50-nuimage-cbgs.pth'),
            '--candidate-mode', 'full', '--fresh-res', '--world-tuning-scope', 'full',
            '--depth-loss-weight', '.1', '--batch-size', '6', '--grad-accum', '2', '--workers', '4',
            '--epochs', '2' if smoke else '10', '--precision', 'bf16', '--optimizer-profile', 'differential',
            '--lr', '1e-4', '--image-lr', '5e-6', '--pretrained-lr', '1e-5',
            '--planner-lr', '2e-5', '--future-lr', '1e-4', '--offset-lr', '1e-4',
            '--world-lr', '1e-6', '--world-readout-steps', '5', '--loss-diagnostics',
            '--weight-decay', '1e-4', '--ddim-steps', '1', '--points-per-map', '9', '9', '16', '25', '36',
            '--window-m', '6', '3', '--offset-mode', 'learned', '--future-dropout', '.1',
            '--monitor-every', '4' if smoke else '100', '--log-every', '2' if smoke else '20',
            '--verify-ddp', '--seed', '2026', '--warmup-steps', '2' if smoke else '500',
            '--stop-after-epoch', str(stop_epoch)]
        if smoke:
            args.extend(['--steps-per-epoch', '8'])
        if stop_epoch > 1:
            args.extend(['--resume', str(work / 'last.pth')])
        return args

    def audit(work, steps, smoke=False):
        config = read(work / 'config.json')
        actual = {g['name']: g['peak_lr'] for g in config['training_optimization']['optimizer_groups']}
        assert actual == RATES, actual
        assert config['architecture_version'] == 10 and config['candidate_config']['mode'] == 'full'
        assert config['nominal_effective_batch'] == 48
        assert config['training_scope'] == 'geobev_candidate_readout_full_irwm'
        assert config['world_tuning_config']['parameter_scope'] == 'all_registered_irwm_parameters'
        assert config['lr_schedule']['total_updates'] == (8 if smoke else 5870)
        assert config['lr_schedule']['warmup_updates'] == (2 if smoke else 500)
        assert config['training_optimization']['world_activation_checkpointing'] is True
        assert config['config']['fresh_res'] is True
        assert not config['config']['warm_start_geobev'] and not config['config']['warm_start']
        assert config['warm_start_audit'] is None
        fresh = config['fresh_initialization_audit']
        assert fresh['status'] == 'PASS' and fresh['prior_res_checkpoint_loaded'] is False
        assert not fresh['optimizer_restored'] and not fresh['scheduler_restored']
        assert fresh['world_trainable_parameters'] == read(IMPL/'real_checks.json')['irwm_total_parameters']
        events = [json.loads(line) for line in (work / 'monitoring.jsonl').read_text().splitlines()]
        assert events and all(math.isfinite(v) for e in events for v in e['metrics'].values())
        for key in ('geobev', 'current_readout', 'future_distance', 'world_tuned',
                    'world_backbone', 'world_neck', 'world_bev_encoder', 'world_future_decoder'):
            assert any(e['metrics'].get('gradient/' + key + '_norm', 0) > 0 for e in events), key
        logs = []
        for line in (work / 'train.log').read_text().splitlines():
            if line.startswith('{'):
                event = json.loads(line)
                if 'loss_total_global' in event:
                    logs.append(event)
        assert logs[-1]['optimizer_updates'] == steps
        assert all(math.isfinite(e[k]) for e in logs for k in ('loss_total_global', 'x0_mse_global'))
        return dict(status='PASS', peak_learning_rates=actual, optimizer_updates=steps,
                    last_train_event=logs[-1], gradient_checks='PASS', config=str(work / 'config.json'))

    def evaluate(checkpoint, epoch):
        evaluation = OUT / ('validation/epoch_%02d' % epoch)
        run('validation_epoch%02d' % epoch, [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=4', str(ROOT / 'tools/eval_geobev_full_coverage.py'),
            '--adapter-checkpoint', str(checkpoint), '--output-dir', str(evaluation),
            '--seed', '2026', '--ddim-steps', '1', '--workers', '2', '--prefetch-factor', '2',
            '--log-seconds', '30'], OUT / ('validation/epoch_%02d.log' % epoch))
        metrics = read(evaluation / 'metrics.json')
        assert metrics['sample_coverage_check'] == 'PASS' and not metrics['partial']
        assert metrics['architecture_version'] == 10 and metrics['adapter_sha256'] == digest(checkpoint)
        assert metrics['seen_frames'] == 6019 and metrics['scored_frames'] == 5119
        assert metrics['rolling_frames'] == 4585 and metrics['fallback_frames'] == 534
        assert metrics['discarded_incomplete_future_frames'] == 900
        assert math.isfinite(metrics['avg_L2_m'])
        return metrics

    def report(values, reference):
        best = min(values, key=lambda v: v['rolling_4585_L2_m'])
        write(OUT / 'epoch_metrics.json', values)
        write(OUT / 'best.json', best)
        lines = ['# V10 从头完整训练：分组学习率与 IRWM 全量更新', '',
            '原始 CVCI2、GeoBEV、IRWM 预训练初始化，V10/Res 新增模块随机初始化；无 V9/A9/V8 权重。',
            '10 epoch / 5870 更新，有效批量 48；500 更新预热后 cosine 衰减到峰值 0.001 倍。',
            '图像骨干 5e-6、原感知 1e-5、DiT 2e-5、新模块 1e-4；IRWM 全部参数 1e-6。',
            '每轮全验证集评估，以原 4585 条滚动集合的 prefix 平均 L2 选最佳，另列含冷启动的 5119 条结果。', '',
            '| Epoch | 原 4585 条 L2/m | 含冷启动 5119 条 L2/m |', '|---|---:|---:|']
        lines += ['| %d | %.9f | %.9f |' % (v['epoch'], v['rolling_4585_L2_m'],
                  v['full_valid_5119_L2_m']) for v in values]
        lines += ['', '当前最佳 epoch %d；原集合 %.9f m，相对 V9 的 %.9f m 变化 %+.9f m。' %
                  (best['epoch'], best['rolling_4585_L2_m'], reference['avg_L2_m'],
                   best['rolling_4585_L2_m']-reference['avg_L2_m']), '',
                  '冷启动不足三帧时用同权重端到端输出；未来六点任一点无效的 900 帧舍弃。',
                  '单种子，相对上一版同时调整学习率与世界模型解冻范围，不能单独归因。', '']
        (OUT / '训练进度.md').write_text('\n'.join(lines), encoding='utf-8')
        return best

    try:
        status('preflight')
        unchanged()
        verification = read(IMPL / 'real_checks.json')
        assert verification['status'] == 'PASS'
        assert shutil.disk_usage(OUT).free > 50 * 1024**3
        assert not (OUT / 'run').exists() and not (OUT / 'checks').exists()
        reference = read(PREVIOUS / 'best.json')  # Comparison metadata only; no model load.
        write(OUT / 'training_recipe.json', dict(architecture_version=10, candidate_mode='full', world_tuning_scope='full',
            initialization='Original CVCI2/GeoBEV/IRWM pretraining; new Res and V10 modules initialized from scratch',
            initialization_checkpoints={n: dict(path=str(MODELS/n),sha256=digest(MODELS/n)) for n in
                ('best_endpoint_1step.pth','MMO_MSO_with_plan_tightly_coupled.pth','geobev-r50-nuimage-cbgs.pth')},
            prior_experiment_checkpoint_loaded=False, epochs=10, total_updates=5870, warmup_updates=500,
            scheduler='linear_warmup_cosine_to_0.001_of_peak', peak_learning_rates=RATES,
            effective_batch=48, batch_per_gpu=6, gradient_accumulation=2, gpu_count=4, world_activation_checkpointing=True,
            precision='BF16 planner/GeoBEV; FP32 TF32 world', seed=2026,
            validation='all val frames; cold E2E fallback; discard incomplete six-point futures',
            note='Restored differential peak rates; every IRWM parameter is trainable at 1e-6, with three-frame and five-step gradients.'))
        smoke = OUT / 'checks/production_batch'
        run('production_batch_check', command(smoke, smoke=True), smoke / 'train.log')
        first_check = audit(smoke, 4, smoke=True)
        run('production_resume_check', command(smoke, stop_epoch=2, smoke=True), smoke / 'train.log', append=True)
        second_check = audit(smoke, 8, smoke=True)
        assert read(smoke/'resume_state_audit.json')['status'] == 'PASS'
        write(OUT / 'production_batch_check.json', dict(status='PASS', initial=first_check,
            resumed=second_check, exact_resume='PASS', effective_batch=48))
        # Reinitialize from the three original pretrained sources after the batch check.
        work = OUT / 'run'
        values = []
        for epoch in range(1, 11):
            run('training_epoch%02d' % epoch, command(work, stop_epoch=epoch), work / 'train.log', append=True)
            train_audit = audit(work, 587*epoch)
            if epoch == 1:
                assert (work/'initial_state.pth').exists()
                assert (read(work/'config.json')['fresh_initialization_audit']['fresh_res_state_sha256'] ==
                        read(smoke/'config.json')['fresh_initialization_audit']['fresh_res_state_sha256'])
            else:
                assert read(work/'resume_state_audit.json')['status'] == 'PASS'
            write(OUT / 'training_audit.json', train_audit)
            checkpoint = work / ('epoch_%02d.pth' % epoch)
            metrics = evaluate(checkpoint, epoch)
            values.append(dict(epoch=epoch, checkpoint=str(checkpoint), sha256=metrics['adapter_sha256'],
                rolling_4585_L2_m=metrics['retained_original_subset']['avg_L2_m'],
                full_valid_5119_L2_m=metrics['avg_L2_m'], train_x0=train_audit['last_train_event']['x0_mse_global']))
            best = report(values, reference)
            status('epoch_complete', completed_epoch=epoch, best_epoch=best['epoch'],
                   best_rolling_L2_m=best['rolling_4585_L2_m'])
        write(OUT / 'result.json', dict(status='PASS', best=best, epoch_metrics=values,
            reference_v9=reference, prior_experiment_checkpoint_loaded=False,
            limitations='Single seed; structure and learning rate changed together.'))
        unchanged()
        status('complete', best_epoch=best['epoch'], best_rolling_L2_m=best['rolling_4585_L2_m'],
               result=str(OUT / 'result.json'))
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
