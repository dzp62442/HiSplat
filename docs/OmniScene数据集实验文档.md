# OmniScene 数据集实验计划

## 1. 配置如何设置与加载
- **数据集配置**：在 `config/dataset/omniscene.yaml` 新增配置，沿用 depthsplat 的字段取值：`defaults.view_sampler=all`（保证任意阶段都获取全部视角），`image_shape=[224, 400]`、`background_color=[0,0,0]`、`name=omniscene`、`roots=[datasets/omniscene]`、`make_baseline_1=false`、`near=0.5`、`far=100.`、`baseline_scale_bounds=false`、`train_times_per_scene=1`、`highres=false`。由于当前 HiSplat 的 `DatasetRE10kCfg` 不包含 `train_times_per_scene/highres`，需要新增 `DatasetOmniSceneCfg` 数据类并在 `src/dataset/__init__.py` 中把 `"omniscene"` 映射到新的 `DatasetOmniScene` 实现。
- **实验配置**：仿照 `config/experiment/re10k.yaml` 同时创建 `config/experiment/omniscene_224x400.yaml` 与 `config/experiment/omniscene_112x200.yaml`，两个文件只在 `dataset.image_shape` 等尺寸相关字段上有所区别；其它内容（`model/encoder: costvolume_pyramid`、`model/decoder: splatting_cuda`、`loss: [mse, lpips]`、`optimizer`/`checkpointing`）保持与 re10k 一致。需要在实验里设置 `data_loader.train.batch_size=1`、`trainer.max_steps=100_001` 并把 `trainer.val_check_interval` 调整为 `0.01`（Lightning 将其解析为相对步数，约等于每 1k step 进行一次验证）；测试阶段继续使用 `test.compute_scores=true`/`test.eval_time_skip_steps=5` 的 re10k 默认值，并在 README 命令中像 depthsplat 那样通过 CLI 传入 `trainer.val_check_interval=0.01` 与 `output_dir=...`。
- **训练/验证/测试节奏**：深度版本使用 `trainer.max_steps=100000`、`train.eval_model_every_n_val=10`（约 10k step 触发一次完整测试）。HiSplat 主程目前没有周期性测试的逻辑，因此我们在训练阶段只依赖 `trainer.val_check_interval=0.01` 的验证频率；若需要测试结果，统一在训练结束后手动运行 `python -m src.main +experiment=omniscene_224x400 mode=test checkpointing.load=...`。`data_loader.test.batch_size`、`data_loader.val.batch_size` 与 `test.compute_scores` 继续沿用 re10k 默认值（1），即可复现 depthsplat 在 OmniScene 上的迭代节奏。

## 2. 数据集如何加载
- **总体流程**：将 depthsplat 的 `src/dataset/dataset_omniscene.py`、`src/dataset/utils_omniscene.py` 引入到 HiSplat，并结合本仓现有的 `DatasetRE10k` 模板调整接口：
  - `__init__`：根据 Stage 加载不同的 bin list（train/val/test/demo），并记录 `self.reso`、`self.near/far`；train/val/test 的抽样策略（train 全量、val 取前 30000 个每隔 3000 采样、test 默认 mini-test / 每 14 个取一个）保持与 depthsplat 一致，以保证对比公平。
  - `__getitem__`：读取 `bin_infos_3.2m/<token>.pkl`，从 6 个环视相机提取 key-frame 作为 context，再额外取每个相机的第 `[1,2]` 帧作为 target；将输入帧拼回 target 末尾，与 depthsplat 的 6→18 帧策略一致。
  - `load_conditions`：复用 depthsplat 的实现，加载 `samples_small`/`sweeps_small` 图像、对应的 parameter JSON、动态掩码（outputs 使用真实 mask，inputs 使用全白 mask），并在 resize 后归一化内参。
  - `context/target` 字段：返回 `extrinsics`（c2w）、`intrinsics`、`image`、`near`、`far`、`index`，并为 target 额外添加 `masks`（`bool`）。这些键需要与 HiSplat 的 `BatchedViews` 类型保持一致。
- **与原有 loader 的差异**：HiSplat 当前的 `DatasetRE10k` 通过 chunk + `ViewSampler` 的方式随机采样 view 组合；OmniScene 则是显式拆分“6 个输入 / 12+6 个输出”，不需要再调用 `ViewSampler.sample`。为兼容接口，可以像 depthsplat 那样把 `ViewSampler` 传入但不使用，或在 `get_dataset` 中检测 `cfg.name == "omniscene"` 后跳过 `view_sampler.sample`。同时需要新增 `DatasetOmniSceneCfg` 并在 `get_dataset` 内注册。
- **可否直接复用 depthsplat 的实现**：读取 bin、加载图像/掩码/相机的逻辑可以直接迁移，但需结合 HiSplat 的差异做以下改动：
  1. `src/dataset/types.py` 目前没有 `masks` 字段，需要参照 depthsplat 扩充 `BatchedViews`/`UnbatchedViews`；
  2. `src/dataset/shims/patch_shim.py` 只裁剪 `image` 和 `intrinsics`，需要像 depthsplat 一样在裁剪时同步裁剪 `masks`，否则动态掩码会错位；
  3. HiSplat 的 `get_view_sampler` 会在 val/test 阶段强制替换成 evaluation sampler（读取 `assets/evaluation_index_*`），但 OmniScene 需要保留 `all` 策略，需在 `get_view_sampler` 中添加 dataset 名检查以跳过该覆盖；
  4. Depthsplat loader 默认读取 `samples_param_small`/`samples_mask_small`，需确认 HiSplat 的部署目录结构与 depthsplat 一致（`datasets/omniscene/...`）；若路径不同，需要在 `load_conditions` 中引入 cfg 字段以自定义前缀。

## 3. 主程序如何调用
- **数据模块**：`src/main.py` 通过 `DataModule` 加载数据。新增 OmniScene 后，`get_dataset` 会返回新的 `DatasetOmniScene`；`LightningDataModule` 的 train/val/test 接口无需改动，只需保证新的数据集返回结构与现有 batch format（`context`/`target` 字典）一致。
- **模型与损失**：HiSplat 的 `ModelWrapper.training_step` 目前直接对 `batch["target"]["image"]` 计算 MSE/LPIPS。若要复用 depthsplat 的动态掩码，需要：
  1. 在 `TrainCfg` 中增加 `use_dynamic_mask`（默认为 `false`，OmniScene 实验时在 CLI/experiment 中设为 `true`）。
  2. 修改 `LossMse`（以及其它需要掩码的 loss）接口，使其可以接收 `valid_depth_mask` 与 depthsplat 类似地屏蔽动态区域。
  3. 在 `ModelWrapper.training_step` 里检测 `self.train_cfg.use_dynamic_mask`，读取 `batch["target"]["masks"]` 构造掩码后传给 loss。
- **调用差异**：depthsplat 在 `train.eval_model_every_n_val`>0 时会额外构造一个 evaluation dataloader（覆盖 view sampler）。HiSplat 没有这一分支，因此 OmniScene 的测试需要通过外部命令触发；同时 HiSplat 默认在 `test` 阶段把 view sampler 改成 evaluation，需要特别处理 `dataset.name==omniscene` 的情况以维持 `all` 策略。
- **可否直接套用 depthsplat 主程**：不建议。两边的 `ModelWrapper`/`Loss` 接口不同，且 HiSplat 的编码器、wandb 配置、checkpoint 行为都与 depthsplat 不同。我们会在 HiSplat 内按其既有风格新增配置项（例如 `train.use_dynamic_mask`）并在 README 中提供 OmniScene 的训练/测试命令，从而保持仓库风格统一。

## 4. 复用与调整清单
- 可直接迁移：`dataset_omniscene.py`、`utils_omniscene.py`、bin 抽样策略、`load_conditions` 的图像/掩码预处理逻辑。
- 需要适配：`Dataset` 注册、`BatchedViews` 类型、`patch_shim`、`get_view_sampler`（跳过 evaluation 覆盖）、`ModelWrapper`/`Loss` 对动态掩码的支持、README 中的 OmniScene 训练/测试指令。
- 验证节奏：沿用 depthsplat 的 batch size 与步数，通过 HiSplat 的 `trainer.val_check_interval` 控制验证频率；测试在训练结束后独立运行 `mode=test` 命令。

以上步骤实现后，即可在 HiSplat 的 `comp_svfgs` 分支上按 depthsplat 的节奏运行 OmniScene 对比实验，并保持模型/优化器配置与原作者在 re10k 上的默认设置一致。
