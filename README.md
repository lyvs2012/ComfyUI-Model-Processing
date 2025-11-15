# ComfyUI 模型处理器

这是一个功能强大的 ComfyUI 插件，支持模型合并、分片模型融合、LoRA 模型融合和模型量化等多种功能，为 Stable Diffusion 模型管理提供全面的解决方案。

## 功能特性

- **模型合并**：支持合并多个 safetensors 格式的模型文件
- **分片模型融合**：自动检测并合并多分片模型（如 diffusion_pytorch_model-00001-of-00005.safetensors）
- **LoRA 模型融合**：支持将 LoRA 模型直接融合到基础模型中
- **模型量化**：支持 4bit、8bit 和 FP8 等多种量化方式，减小模型体积
- **智能路径处理**：优化的 Windows 路径处理，自动处理 BOM 和不可见字符
- **内存优化**：分片处理机制，支持大模型合并时的内存优化
- **详细日志**：提供完整的执行日志，显示进度、错误和结果信息

## 安装方法

1. 确保已安装 ComfyUI
2. 将此插件目录复制到 ComfyUI 的 `custom_nodes` 文件夹中
3. 安装依赖：
   ```
   pip install safetensors
   ```
4. 重启 ComfyUI

## 核心功能使用说明

### 1. 基本模型合并

**功能说明**：合并多个完整的模型文件。

**参数设置**：
- **operation_type**: 选择 `merge`
- **main_model**: 留空
- **model_files**: 输入模型文件路径，每行一个文件路径
- **output_file**: 设置合并后的模型文件名（如 `merged_model.safetensors`）
- **output_directory**: 设置输出目录路径
- **merge_mode**: 选择合并模式（`update` 或 `replace`）
- **quantize_bits**: 选择量化方式（`none` 表示不量化）

**使用步骤**：
1. 在 ComfyUI 中，从 "模型工具" 类别中找到 "增强模型处理器" 节点，或者直接搜索 "增强模型处理器" 节点
2. 设置下述参数
3. 点击“运行”
4. 查看控制台输出了解合并进度

### 2. 分片模型融合

**功能说明**：自动检测并合并多分片模型文件，如 Hugging Face 下载的大型模型。

**参数设置**：
- **operation_type**: 选择 `merge`
- **main_model**: 留空
- **model_files**: 只需提供第一个分片文件路径（如 `diffusion_pytorch_model-00001-of-00005.safetensors`）
- **output_file**: 设置输出文件名
- **output_directory**: 设置输出目录
- **use_sharded_processing**: 设为 `True`
- **shard_size_mb**: 设置分片大小（建议值：512-4096，单位 MB）
- **auto_adjust_shard**: 设为 `True`（自动根据可用内存调整分片大小）

**自动检测机制**：
插件会自动从第一个分片文件名中提取信息，生成并查找所有其他分片文件。例如，提供第一个分片后，系统会自动查找并处理：
- diffusion_pytorch_model-00002-of-00005.safetensors
- diffusion_pytorch_model-00003-of-00005.safetensors
- diffusion_pytorch_model-00004-of-00005.safetensors
- diffusion_pytorch_model-00005-of-00005.safetensors

### 3. LoRA 模型融合

**功能说明**：将 LoRA 模型的权重融合到基础模型中，生成新的合并模型。

**参数设置**：
- **operation_type**: 选择 `merge`
- **main_model**: 设置基础模型文件路径
- **lora_file**: 设置 LoRA 模型文件路径
- **lora_scale**: 设置 LoRA 权重缩放比例（默认 1.0，范围 0.0-2.0）
- **output_file**: 设置输出文件名
- **output_directory**: 设置输出目录
- **quantize_bits**: 选择是否量化输出模型

**使用技巧**：
- 使用较低的 lora_scale（如 0.7-0.9）可以获得更自然的效果
- 某些 LoRA 可能在特定的缩放比例下效果最佳，建议尝试不同的值

### 4. 模型量化

**功能说明**：减小模型文件体积，适合在显存较小的设备上使用。

**参数设置**：
- **operation_type**: 选择 `quantize`
- **main_model**: 设置需要量化的模型文件路径
- **output_file**: 设置输出文件名
- **output_directory**: 设置输出目录
- **quantize_bits**: 选择量化方式：
  - `none`: 不量化
  - `4bit`: 4位量化，体积最小
  - `8bit`: 8位量化，平衡体积和质量
  - `fp8`: FP8 量化，保持较高精度的同时减小体积

**量化效果对比**：
- 4bit 量化：体积减小约 75%，推理速度可能略有下降
- 8bit 量化：体积减小约 50%，推理速度影响较小
- FP8 量化：体积减小约 50%，精度保持较好

## 高级功能

### 1. 合并模式说明

- **update 模式**：按顺序合并所有模型，后一个模型的权重会覆盖前一个模型相同层的权重
- **replace 模式**：使用最后一个模型完全替换之前的合并结果

### 2. 分片大小优化

- **手动设置**：通过 `shard_size_mb` 参数设置分片大小
- **自动优化**：启用 `auto_adjust_shard` 后，系统会根据可用内存自动调整最佳分片大小
- **建议值**：
  - 16GB 内存：1024-2048 MB
  - 8GB 内存：512-1024 MB
  - 4GB 内存：256-512 MB

## 工作流示例

本插件包含一个示例工作流文件 `Model-Processing模型处理器.json`，你可以在 ComfyUI 中加载此文件来快速开始。

## 注意事项

- **路径格式**：Windows 系统中请使用正确的路径格式（如 `E:\Models\model.safetensors`）
- **文件权限**：确保对输出目录有写入权限
- **内存要求**：合并大型模型时，建议系统内存不低于 16GB
- **文件名格式**：分片模型文件名必须遵循标准格式（如 `name-00001-of-00005.safetensors`）

## 故障排除

### 常见问题解决

1. **文件找不到错误**
   - 检查文件路径是否正确
   - 确保文件名大小写正确
   - Windows 路径中避免使用特殊字符

2. **内存不足错误**
   - 启用分片处理功能
   - 减小 `shard_size_mb` 参数值
   - 关闭其他占用内存的应用程序

3. **LoRA 融合效果不佳**
   - 调整 `lora_scale` 参数值
   - 确保基础模型与 LoRA 模型架构兼容
   - 检查 LoRA 文件是否损坏

4. **量化模型质量问题**
   - 如果量化后质量下降明显，尝试使用更高位的量化方式
   - FP8 量化通常能提供更好的质量平衡

5. **路径中包含不可见字符**
   - 插件已优化处理，自动去除 BOM 标记和不可见字符
   - 如果仍有问题，请重新复制粘贴文件路径

## 版本历史

### 最新版本
- 优化分片模型自动检测功能
- 改进 Windows 路径处理，支持 BOM 和不可见字符处理
- 修复 final_output_path 变量未定义问题
- 增强日志输出信息

主页：https://github.com/lyvs2012/ComfyUI-Model-Processing

