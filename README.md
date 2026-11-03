# CVCI2_Res V10 — GeoBEV + candidate interaction + full IRWM training

本仓库保存 **2026-11-03 正在训练的 V10 全量 IRWM 版本**。源码直接从训练服务器导出，并逐文件核对运行时记录的 SHA256。运行标识：`h20_v10_fresh_fullworld10_20261103_01`。

## 本版本结构

- 当前视觉使用 GeoBEV；IRWM 保留独立的预训练视觉与 BEV 编码通路。
- DiT 为 12 层（6 self + 6 interaction），预测六个未来点的干净位移 `x0`，每层等权监督。
- 交互层依次处理 agent、ego、当前 BEV，再融合经过可靠性门控的未来条件。每层轨迹输出的几何副本更新下一层采样位置。
- 当前 BEV 按轨迹位置取中心、切向前后、法向左右五点；未来条件有位置与距离偏置。运动状态拼接进原门控输入。
- IRWM 每次执行五步未来滚动，读取 `[9, 9, 16, 25, 36]`，共 95 个未来 token。
- **IRWM 全部已加载参数均解冻，峰值学习率 `1e-6`**；梯度贯穿三帧观测编码及五步未来滚动，启用激活重计算。BN 保持预训练运行统计，仿射参数可训练。

## 训练配方

从原始 CVCI2、GeoBEV、IRWM 预训练权重初始化；新增 Res/V10 模块重新初始化。**没有加载已训练的 V9、A9、V8 或 V10 检查点。**

| 优化器参数组 | 峰值学习率 |
|---|---:|
| `image_backbone` | `5e-6` |
| `pretrained_other` | `1e-5` |
| `pretrained_planner` | `2e-5` |
| `new_future` | `1e-4` |
| `new_offset` | `1e-4` |
| `new_bev_bridge` | `1e-4` |
| `geobev_depth_heads` | `1e-5`，weight decay 为 0 |
| `world_tuned`（全部 IRWM） | `1e-6` |
| `new_current_readout` | `1e-4` |

- 4 张 H20-3e；每卡 batch 6、梯度累积 2 次，有效全局 batch 48。
- 10 epoch；原始数据流每轮 1173 微步、587 次更新，总计 5870 次更新。
- 500 次优化更新线性预热，随后 cosine 衰减至各组峰值的 `0.001` 倍。
- fused AdamW；weight decay `1e-4`；全局梯度范数裁剪 `1.0`；seed `2026`。
- GeoBEV / DiT 使用 BF16；IRWM 使用 FP32 / TF32。
- 均匀采样扩散时间步 `0..99`，future dropout `0.1`，推理一次去噪（`t=99`）。
- 总可训练参数 `177,618,274`，其中 IRWM 为 `76,557,099`。

损失：

```text
mean(12 layers of masked x0 MSE)
  + 0.2 * agent classification
  + 0.2 * agent box regression
  + 0.1 * current occupancy
  + 0.1 * GeoBEV depth
```

IRWM 原有部分输出头未被当前规划损失消费；全部参数设为可训练不代表每个输出头都会获得非零规划梯度。本版本未另外添加 IRWM 占据或语义监督。

## 配置在哪里

| 文件 | 用途 |
|---|---|
| [`configs/v10_full_irwm/train.json`](configs/v10_full_irwm/train.json) | 便于复现的完整训练配方 |
| [`configs/v10_full_irwm/recorded_config.json`](configs/v10_full_irwm/recorded_config.json) | 正在训练任务生成的原始完整配置，含真实分组、初始化审计、结构和源码哈希 |
| [`configs/v10_full_irwm/recorded_recipe.json`](configs/v10_full_irwm/recorded_recipe.json) | 原始任务配方及三份预训练权重 SHA256 |
| [`configs/v10_full_irwm/recorded/`](configs/v10_full_irwm/recorded/) | 服务器实际使用的协调脚本与训练包装器，原样保留 |
| [`configs/v10_full_irwm/export_manifest.json`](configs/v10_full_irwm/export_manifest.json) | 从服务器导出时的逐文件摘要 |
| [`environment/`](environment/) | 原始环境锁定文件与 H20 MMCV/CUDA 构建记录 |

`recorded_config.json` 中 `stop_after_epoch=1` 表示协调器每次运行到一轮结束后插入验证；完整计划仍是 **10 epoch / 5870 次更新**。第二轮起仅恢复同一任务的 `run/last.pth`，连续保留优化器、调度器、数据流、历史和随机状态。

## 准备环境、数据与权重

环境准备见 [`environment/README.md`](environment/README.md)。原始环境为 Linux x86_64、Python 3.8.20、Torch 2.0.0+cu118、MMCV-full 1.7.2（为 H20 `sm_90` 编译）和 MMDetection 2.28.2。IRWM 的兼容运行时代码已包含在 `vendor/irwm_runtime/`。

数据目录需包含 nuScenes trainval 图像、点云、元数据，以及项目所用的 `nuscenes_infos_train.pkl`、`nuscenes_infos_val.pkl`。原服务器入口为 `/mnt/workspace/dataSet/nuscene`。点云用于训练深度/占据监督，当前规划视觉入口为六相机。

将以下三份原始权重放入仓库同级 `Models/`，或用 `--models-dir` 指定目录：

```text
Models/
  best_endpoint_1step.pth
  geobev-r50-nuimage-cbgs.pth
  MMO_MSO_with_plan_tightly_coupled.pth
```

本仓库只发布源码、配置和说明；权重、数据集、训练日志与模型输出均单独保存在服务器。

## 启动完整训练

先查看实际命令（不会启动训练或创建输出目录）：

```bash
python3 tools/run_v10_full_irwm.py \
  --data-root /mnt/workspace/dataSet/nuscene \
  --models-dir /mnt/workspace/nix3sgh/Models \
  --output experiments/v10_full_irwm_reproduction \
  --dry-run
```

正式运行时删除 `--dry-run`：

```bash
ENV/bin/python tools/run_v10_full_irwm.py \
  --data-root /mnt/workspace/dataSet/nuscene \
  --models-dir /mnt/workspace/nix3sgh/Models \
  --output experiments/v10_full_irwm_reproduction
```

启动器会核对导出的训练源码和三份原始权重，使用四卡完整训练，每轮结束后验证并保存 `epoch_metrics.json`、`best.json` 和逐轮检查点。输出目录必须是新目录。该可迁移启动器是公开快照附带的复现工具；正在训练任务使用的原始脚本保存在 `recorded/` 中。服务器原任务还在正式训练前完成独立的四卡短训练与精确恢复检查，随后重新初始化正式模型；短检查权重没有用于正式训练。

## 验证协议

遍历全部 6019 条验证标注；未来六点全有效的 5119 帧计分，其中包含 4585 条原滚动集合和 534 条冷启动不足三帧的端到端 fallback。未来点不完整的 900 帧舍弃。

分别保存原滚动集合和含冷启动集合的 1/2/3 秒 prefix L2；以 **原 4585 条集合的三个 prefix L2 平均值**选择最佳轮次。端点 L2 另行报告，不能与 prefix 指标混用。评估脚本为 [`tools/eval_geobev_full_coverage.py`](tools/eval_geobev_full_coverage.py)。

本次导出时完整训练仍在进行，仓库不将短检查损失或旧版本结果作为本版本的最终精度。

## 来源

基础工程：[CVCI2_Res](https://github.com/123mike-source/CVCI2_Res)、[CVCI2](https://github.com/123mike-source/CVCI2)。GeoBEV 派生模块保留其 [`LICENSE`](geobev_encoder/LICENSE) 与 [`NOTICE`](geobev_encoder/NOTICE)。各第三方源码的原有版权和许可声明保持在对应文件中。
