# Recorded H20 environment

本目录保存当前训练环境的记录，不包含虚拟环境或编译二进制。

- Python 3.8.20，Linux x86_64。
- Torch 2.0.0+cu118、torchvision 0.15.1+cu118。
- MMDetection 2.28.2；MMCV-full 1.7.2 按 H20 的 `sm_90` 架构构建。
- NumPy 1.23.5，SciPy 1.10.1，nuScenes-devkit 1.1.10。
- `recorded_requirements.lock.txt`：服务器环境实际 `pip freeze` 记录。
- `recorded_mmcv_sm90_build.json`、`recorded_cuda_toolchain_manifest.json`：编译器/工具链记录，其中绝对路径保留原服务器信息。

## 重建

在独立 Linux 工作目录运行仓库内的原始环境脚本：

```bash
python3 setup_env.py
ENV/bin/python tools/check_environment.py
```

`setup_env.py` 在项目内安装 Python 3.8 与依赖，配置本地 OpenCV 动态库，并调用 `tools/build_mmcv_sm90.py` 构建 MMCV CUDA 算子。工具链脚本会下载 CUDA 11.8 组件和 GCC 11；系统需提供 `apt-get`、`dpkg-deb`、`curl`，以及可用的 NVIDIA 驱动和下载网络。

不要用新版 MMDetection/MMCV 替换 1.x API，也不要另装现代 mmdet3d；IRWM 使用仓库中的兼容运行时。原始 H20 每卡总显存约 140 GiB；本轮全量梯度配置的首批 rank 0 峰值分配显存约 114 GiB。其他显卡或环境需要重新验证批量及算子支持。

完整启动配方见 `configs/v10_full_irwm/train.json` 和仓库首页。附带启动器可以在不导入 Torch、不占用 GPU 的情况下执行 `--dry-run` 检查命令。
