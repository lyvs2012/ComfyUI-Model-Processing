"""
模型处理器节点 - 用于合并多个 safetensors 模型文件、量化主模型和融合LORA模型
功能包括：分片模型融合、模型量化（支持4/8/88位）、主模型融合LORA
"""

import os
import torch
import gc
import time
import psutil
from safetensors.torch import save_file, load_file
from typing import Dict, List, Tuple, Optional

class ModelMergerNode:
    """
    增强型模型处理节点，支持分片模型融合、模型量化和主模型融合LORA模型
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation_type": (["merge", "quantize", "merge_lora"], {
                    "default": "merge",
                    "label": "操作类型",
                    "description": "选择要执行的操作：合并模型、量化模型或融合LoRA"
                }),
                "main_model": ("STRING", {
                    "placeholder": "主模型文件路径"
                }),
                "output_file": ("STRING", {
                    "default": "merged_model.safetensors",
                    "placeholder": "输出的模型文件名"
                }),
            },
            "optional": {
                # 合并模型相关参数
                "model_files": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入模型文件路径，每行一个（合并模式下使用）"
                }),
                "merge_mode": (["update", "replace"], {
                    "default": "update",
                    "label": "合并模式",
                    "description": "更新模式:更新现有键值，替换模式:完全替换"
                }),
                "use_sharded_processing": ("BOOLEAN", {
                    "default": True,
                    "label": "使用分片处理",
                    "description": "对大型模型使用分片处理以减少内存使用"
                }),
                "shard_size_mb": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                    "label": "分片大小(MB)",
                    "description": "每个分片的大致大小，单位为MB"
                }),
                "auto_adjust_shard": ("BOOLEAN", {
                    "default": True,
                    "label": "自动调整分片大小",
                    "description": "根据可用内存自动调整最佳分片大小"
                }),
                
                # 量化相关参数
                "quantize_bits": (["none", "4bit", "8bit", "fp8_e4m3fn"], {
                    "default": "none",
                    "label": "量化方式",
                    "description": "选择模型量化方式：无(不量化)、4位量化、8位量化、FP8量化(e4m3fn格式)"
                }),
                
                # LORA融合相关参数
                "lora_file": ("STRING", {
                    "placeholder": "LORA模型文件路径（融合模式下使用）"
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "LORA融合比例"
                }),
                
                # 通用参数
                "output_directory": ("STRING", {
                    "default": "",
                    "placeholder": "输出模型保存目录"
                }),
                "overwrite_existing": ("BOOLEAN", {
                    "default": True,
                    "label": "覆盖现有文件",
                    "description": "如果目标文件已存在，是否覆盖"
                }),
                "enable_detailed_logging": ("BOOLEAN", {
                    "default": False,
                    "label": "启用详细日志",
                    "description": "输出更详细的处理信息"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "process_model"
    CATEGORY = "模型工具"
    DISPLAY_NAME = "增强模型处理器"
    
    def __init__(self):
        """初始化模型处理器"""
        self.supported_extensions = [".safetensors", ".lora"]
        self.common_model_dirs = [
            os.getcwd(),
            os.path.join(os.getcwd(), 'models'),
            os.path.join(os.getcwd(), 'models', 'checkpoints'),
            os.path.join(os.getcwd(), 'models', 'lora')
        ]
        self.max_memory_ratio = 0.7  # 使用最多70%的可用内存
        self.min_shard_size_mb = 128  # 最小分片大小
        self.max_shard_size_mb = 4096  # 最大分片大小
    
    def _normalize_path(self, path: str) -> str:
        """规范化文件路径，处理Windows路径格式问题"""
        if not path:
            return path
        
        # 移除可能的BOM标记和不可见字符
        # 特别是处理Windows复制粘贴时可能引入的特殊字符
        path = path.lstrip('\ufeff\ufeff\ufeff\ufeff\ufeff\ufeff')
        
        # 去除首尾空白
        path = path.strip()
        
        # 处理转义序列问题
        if '\\\\' in path:
            path = path.replace('\\\\', '\\')
        
        # 统一路径分隔符
        path = path.replace('/', '\\')
        
        # 修复驱动器路径格式
        if len(path) > 2 and path[1] == ':' and path[2] != '\\':
            path = path[:2] + '\\' + path[2:]
        
        # 确保路径没有其他不可见字符
        path = ''.join(char for char in path if ord(char) >= 32 or char in '\t\n\r')
        
        return path
    
    def _get_potential_paths(self, base_path: str) -> List[str]:
        """获取可能的文件路径变体列表"""
        variants = [base_path]  # 原始路径
        
        # 添加绝对路径
        try:
            abs_path = os.path.abspath(base_path)
            variants.append(abs_path)
        except Exception:
            pass
        
        # 修复驱动器号问题
        if len(base_path) >= 2:
            if base_path[1] != ':' and base_path[0].isalpha():
                drive_path = f"{base_path[0]}:{base_path[1:]}"
                variants.append(drive_path)
            elif base_path[1] == ':' and len(base_path) > 2 and base_path[2] != '\\':
                fixed_drive = f"{base_path[0]}:{base_path[2:]}"
                variants.append(fixed_drive)
        
        # 添加当前工作目录路径
        cwd_path = os.path.join(os.getcwd(), os.path.basename(base_path))
        variants.append(cwd_path)
        
        # 添加可能的扩展名
        file_name = os.path.basename(base_path)
        name_without_ext = os.path.splitext(file_name)[0]
        
        for ext in self.supported_extensions:
            if not file_name.lower().endswith(ext):
                variants.append(os.path.join(os.path.dirname(base_path), name_without_ext + ext))
        
        # 在常见模型目录中查找
        for common_dir in self.common_model_dirs:
            try:
                variants.append(os.path.join(common_dir, file_name))
            except Exception:
                pass
        
        # 去重
        return list(dict.fromkeys(variants))
    
    def _find_file(self, file_path: str) -> str:
        """查找文件，如果找不到则抛出异常"""
        # 获取可能的路径变体
        potential_paths = self._get_potential_paths(self._normalize_path(file_path))
        
        # 尝试所有变体
        for path in potential_paths:
            if os.path.exists(path):
                print(f"找到文件: {path}")
                return path
        
        # 如果没找到，尝试搜索常见目录
        file_name = os.path.basename(file_path)
        for common_dir in self.common_model_dirs:
            if os.path.exists(common_dir):
                for root, _, files in os.walk(common_dir, topdown=False):
                    for name in files:
                        if name.lower() == file_name.lower() or any(name.lower() == file_name.lower() + ext for ext in self.supported_extensions):
                            found_path = os.path.join(root, name)
                            print(f"搜索到文件: {found_path}")
                            return found_path
        
        # 如果仍然找不到，抛出异常
        raise FileNotFoundError(
            f"文件不存在: {file_path}\n"
            f"请检查路径格式是否正确（Windows路径示例: C:\\Models\\model.safetensors）\n"
            f"文件扩展名是否完整（.safetensors或.lora）"
        )
    
    def _prepare_output_path(self, output_directory: str, output_file: str) -> str:
        """准备输出文件路径"""
        # 规范化输出目录
        if not output_directory:
            output_directory = os.path.join(os.getcwd(), "output")
        
        output_directory = self._normalize_path(output_directory)
        
        # 确保输出目录存在
        os.makedirs(output_directory, exist_ok=True)
        
        # 构建完整输出路径
        output_path = os.path.join(output_directory, output_file)
        
        # 确保扩展名正确
        if not output_path.lower().endswith('.safetensors'):
            output_path += '.safetensors'
        
        return output_path
    
    def _get_optimal_shard_size(self, target_shard_size_mb: int) -> int:
        """根据可用内存自动计算最佳分片大小"""
        try:
            # 获取系统可用内存（MB）
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            print(f"[分片优化] 可用内存: {available_memory_mb:.2f} MB")
            
            # 计算推荐的分片大小（使用最大可用内存的一部分）
            recommended_shard_size = available_memory_mb * self.max_memory_ratio
            print(f"[分片优化] 推荐分片大小: {recommended_shard_size:.2f} MB")
            
            # 限制在合理范围内
            optimal_shard_size = max(
                self.min_shard_size_mb,
                min(recommended_shard_size, self.max_shard_size_mb)
            )
            
            # 四舍五入到最接近的64MB倍数
            optimal_shard_size = round(optimal_shard_size / 64) * 64
            
            print(f"[分片优化] 优化后的分片大小: {optimal_shard_size} MB")
            return optimal_shard_size
        except Exception as e:
            print(f"[分片优化] 计算最佳分片大小失败: {str(e)}，使用默认值")
            return target_shard_size_mb
    
    def _get_memory_usage(self) -> float:
        """获取当前Python进程的内存使用情况（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _log_progress(self, current: int, total: int, operation: str, detailed: bool = False) -> None:
        """记录进度信息"""
        progress_percent = (current / total) * 100 if total > 0 else 0
        memory_usage = self._get_memory_usage()
        
        base_log = f"[进度] {operation} {progress_percent:.1f}% - 内存使用: {memory_usage:.2f} MB"
        
        if detailed:
            print(base_log)
        else:
            # 每10%输出一次
            if current == 1 or current == total or (current % (total // 10 if total >= 10 else 1) == 0):
                print(base_log)
    
    def _sharded_merge(self, model_files: List[str], output_path: str, merge_mode: str, 
                      shard_size_mb: int, auto_adjust: bool = True, detailed_logging: bool = False) -> str:
        """使用分片处理合并大型模型，支持自动调整分片大小和详细进度跟踪"""
        print(f"使用分片合并，初始分片大小: {shard_size_mb}MB")
        
        # 自动调整分片大小
        if auto_adjust:
            shard_size_mb = self._get_optimal_shard_size(shard_size_mb)
        
        # 初始化合并后的字典
        merged_dict = {}
        total_keys_processed = 0
        
        # 计算分片大小（以字节为单位）
        shard_size_bytes = shard_size_mb * 1024 * 1024
        
        # 记录开始时间
        start_time = time.time()
        total_files = len(model_files)
        
        # 预读取文件大小信息，用于进度显示
        file_sizes = []
        for file_path in model_files:
            try:
                file_sizes.append(os.path.getsize(file_path))
            except Exception:
                file_sizes.append(0)
        
        for file_idx, file_path in enumerate(model_files, 1):
            print(f"处理文件 {file_idx}/{total_files}: {file_path}")
            
            # 加载当前模型文件
            current_dict = load_file(file_path)
            current_keys = list(current_dict.keys())
            file_key_count = len(current_keys)
            
            # 分批处理键值对
            current_shard = {}
            current_shard_size = 0
            shard_count = 0
            
            for key_idx, (key, tensor) in enumerate(current_dict.items(), 1):
                # 估算张量大小
                tensor_size = tensor.nelement() * tensor.element_size()
                
                # 检查是否需要创建新分片
                if current_shard_size + tensor_size > shard_size_bytes and current_shard:
                    # 处理当前分片
                    self._process_shard(current_shard, merged_dict, merge_mode, shard_count, detailed_logging)
                    
                    # 清除当前分片并释放内存
                    current_shard.clear()
                    current_shard_size = 0
                    shard_count += 1
                    gc.collect()
                    
                    # 记录进度
                    total_processed = total_keys_processed + key_idx
                    self._log_progress(total_processed, sum(file_key_count for file_key_count in [len(load_file(f)) for f in model_files]), 
                                      f"文件 {file_idx}/{total_files}, 分片 {shard_count}", detailed_logging)
                
                # 添加到当前分片
                current_shard[key] = tensor
                current_shard_size += tensor_size
                
                # 定期释放未使用的张量
                if key_idx % 100 == 0:
                    gc.collect()
            
            # 处理剩余的键值对
            if current_shard:
                self._process_shard(current_shard, merged_dict, merge_mode, shard_count, detailed_logging)
                current_shard.clear()
                gc.collect()
            
            total_keys_processed += file_key_count
            
            # 释放当前模型文件的内存
            current_dict.clear()
            gc.collect()
            
            # 文件处理完成，记录进度
            self._log_progress(file_idx, total_files, "文件处理", detailed_logging)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        print(f"分片合并完成，共处理 {total_keys_processed} 个键值对，耗时 {elapsed_time:.2f} 秒")
        
        # 保存合并后的模型
        final_output_path = self._save_model_safely(merged_dict, output_path)
        
        # 释放内存
        merged_dict.clear()
        gc.collect()
        
        return final_output_path
    
    def _process_shard(self, current_shard: Dict, merged_dict: Dict, merge_mode: str, 
                      shard_index: int, detailed_logging: bool = False) -> None:
        """处理单个分片的合并逻辑"""
        if detailed_logging:
            print(f"[分片处理] 处理分片 {shard_index}，包含 {len(current_shard)} 个键值对")
        
        # 根据合并模式更新字典
        if merge_mode == "update":
            merged_dict.update(current_shard)
        else:  # replace模式
            merged_dict.clear()
            merged_dict.update(current_shard)
    
    def _save_model_safely(self, state_dict: Dict, output_path: str) -> str:
        """
        安全保存模型，增强Windows兼容性，使用shutil.move替代os.rename，并添加更多错误处理
        返回实际保存的文件路径
        """
        print(f"开始保存模型到: {output_path}")
        start_time = time.time()
        import shutil
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"创建输出目录: {output_dir}")
            except Exception as e:
                print(f"创建输出目录失败: {str(e)}")
                raise RuntimeError(f"创建输出目录失败: {str(e)}")
        
        # 检查路径是否有效
        if not output_path or not isinstance(output_path, str):
            raise ValueError(f"无效的输出路径: {output_path}")
        
        # 检查文件权限
        try:
            # 测试写入权限
            test_path = os.path.join(output_dir, "test_write_permission.tmp")
            with open(test_path, 'wb') as f:
                f.write(b'test')
            os.remove(test_path)
            print("文件写入权限检查通过")
        except Exception as e:
            print(f"警告: 文件写入权限检查失败: {str(e)}")
            print(f"输出目录: {output_dir}")
            print(f"当前用户: {os.getlogin() if hasattr(os, 'getlogin') else '未知'}")
        
        # 方法1: 直接保存
        try:
            # 直接保存
            save_file(state_dict, output_path)
            elapsed = time.time() - start_time
            print(f"成功保存模型至: {output_path}，耗时 {elapsed:.2f} 秒")
            return output_path
        except IOError as e:
            print(f"直接保存失败: {str(e)}")
            print("尝试使用临时文件方法保存...")
        except Exception as e:
            print(f"直接保存发生其他错误: {str(e)}")
            print("尝试使用临时文件方法保存...")
        
        # 方法2: 使用临时文件方法（使用不同的临时文件路径）
        temp_path = output_path + f".tmp_{int(time.time())}"
        try:
            print(f"尝试保存到临时文件: {temp_path}")
            save_file(state_dict, temp_path)
            print("临时文件保存成功")
            
            # 如果原文件存在，先尝试删除
            if os.path.exists(output_path):
                try:
                    print(f"尝试删除原文件: {output_path}")
                    os.remove(output_path)
                    print("原文件删除成功")
                except Exception as e:
                    print(f"删除原文件失败: {str(e)}")
                    print("尝试使用shutil.move强制覆盖...")
            
            # 使用shutil.move替代os.rename，因为shutil.move在Windows上有更好的兼容性
            print(f"使用shutil.move移动临时文件到目标路径")
            shutil.move(temp_path, output_path)
            
            elapsed = time.time() - start_time
            print(f"通过临时文件方法成功保存至: {output_path}，耗时 {elapsed:.2f} 秒")
            return output_path
        except Exception as e2:
            print(f"临时文件方法保存失败: {str(e2)}")
            # 尝试删除临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print("临时文件已清理")
                except:
                    print("警告: 无法清理临时文件")
        
        # 方法3: 尝试使用不同的文件名
        try:
            backup_path = output_path.replace('.safetensors', f'_backup_{int(time.time())}.safetensors')
            print(f"尝试使用备用文件名保存: {backup_path}")
            save_file(state_dict, backup_path)
            elapsed = time.time() - start_time
            print(f"成功保存到备用路径: {backup_path}，耗时 {elapsed:.2f} 秒")
            # 提示用户
            print(f"\n重要提示: 由于原始路径保存失败，模型已保存到备用路径")
            print(f"请手动将 {backup_path} 重命名为 {os.path.basename(output_path)}")
            return backup_path  # 返回备用路径
        except Exception as e3:
            print(f"备用文件名保存也失败: {str(e3)}")
        
        # 所有方法都失败，抛出异常
        raise RuntimeError(f"保存模型失败，已尝试多种方法。请检查文件权限和磁盘空间。")
    
    def _quantize_model(self, main_model: str, output_path: str, bits: str) -> str:
        """量化模型到指定的方式，支持4位、8位和fp8_e4m3fn量化格式"""
        print(f"开始量化模型，目标方式: {bits}")
        print(f"主模型路径: {main_model}")
        print(f"输出路径: {output_path}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 根据量化位数设置策略
        quant_strategy = self._get_quantization_strategy(bits)
        print(f"量化策略: {quant_strategy['name']}")
        print(f"目标数据类型: {quant_strategy['target_dtype']}")
        
        # 加载模型
        try:
            print("正在加载模型...")
            state_dict = load_file(main_model)
            print(f"模型加载完成，包含 {len(state_dict)} 个键值对")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise RuntimeError(f"加载模型失败: {str(e)}")
        
        # 初始化量化后的字典
        quantized_dict = {}
        total_keys = len(state_dict)
        processed_float_tensors = 0
        skipped_tensors = 0
        
        # 预处理：计算张量统计信息
        tensor_stats = self._analyze_tensors(state_dict)
        print(f"张量分析完成: {tensor_stats['float_tensors']} 个浮点张量，{tensor_stats['non_float_tensors']} 个非浮点张量")
        
        # 进行量化
        for key_idx, (key, tensor) in enumerate(state_dict.items(), 1):
            try:
                # 只量化浮点类型的张量
                if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    # 如果是bfloat16类型，先转换为float32以获得更好的量化效果
                    original_dtype = tensor.dtype
                    if original_dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    processed_float_tensors += 1
                    
                    # 记录张量信息
                    if key_idx % 100 == 0:
                        print(f"处理张量 {key_idx}/{total_keys}: {key}")
                        print(f"  - 形状: {tensor.shape}, 类型: {tensor.dtype}")
                        print(f"  - 数据范围: {tensor.min().item():.6f} 到 {tensor.max().item():.6f}")
                    
                    # 根据量化方式进行处理
                    if bits == "8bit":
                        # 8位量化 - 更精确的实现
                        quantized_tensor = self._quantize_to_8bit(tensor)
                    elif bits == "4bit":
                        # 4位量化 - 改进的实现
                        quantized_tensor = self._quantize_to_4bit(tensor)
                    elif bits == "fp8_e4m3fn":
                        # fp8_e4m3fn量化格式
                        quantized_tensor = self._quantize_to_fp8(tensor)
                    else:
                        # 不量化，保持原类型
                        quantized_tensor = tensor
                        
                    # 添加到量化字典
                    quantized_dict[key] = quantized_tensor
                else:
                    # 非浮点类型保持不变
                    quantized_dict[key] = tensor
                    skipped_tensors += 1
                    
                # 记录进度并定期释放内存
                if key_idx % 100 == 0:
                    progress = (key_idx / total_keys) * 100
                    memory_usage = self._get_memory_usage()
                    print(f"量化进度: {progress:.1f}% - 已处理 {key_idx}/{total_keys} 个键值对")
                    print(f"  - 浮点张量: {processed_float_tensors} 个已处理")
                    print(f"  - 内存使用: {memory_usage:.2f} MB")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"处理张量 {key} 时出错: {str(e)}")
                print(f"  - 错误详情: {repr(e)}")
                # 出错时使用原始张量
                quantized_dict[key] = tensor
                skipped_tensors += 1
        
        # 计算量化统计信息
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print(f"量化操作完成")
        print(f"- 总键值对: {total_keys}")
        print(f"- 处理的浮点张量: {processed_float_tensors}")
        print(f"- 跳过的张量: {skipped_tensors}")
        print(f"- 量化位数: {bits}")
        print(f"- 总耗时: {elapsed_time:.2f} 秒")
        print(f"- 平均每个张量处理时间: {(elapsed_time / max(1, processed_float_tensors)):.3f} 秒")
        print("=" * 80)
        
        # 保存量化后的模型
        final_output_path = output_path
        try:
            # 调用保存方法，可能返回备用路径
            result = self._save_model_safely(quantized_dict, output_path)
            # 如果返回了备用路径，使用它
            if result and result != output_path:
                final_output_path = result
        except Exception as e:
            error_msg = f"保存量化模型失败: {str(e)}"
            print(error_msg)
            print("建议解决方案:")
            print("1. 检查输出目录是否有写入权限")
            print("2. 确保磁盘有足够空间")
            print("3. 尝试使用不同的输出目录")
            print("4. 关闭可能正在占用该文件的其他程序")
            raise RuntimeError(error_msg)
        finally:
            # 确保释放内存
            print("执行最终内存回收...")
            if 'state_dict' in locals():
                state_dict.clear()
            if 'quantized_dict' in locals():
                quantized_dict.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        return final_output_path
    
    def _get_quantization_strategy(self, bits: str) -> Dict:
        """根据量化方式获取对应的量化策略"""
        strategies = {
            "none": {
                "name": "不量化",
                "target_dtype": "original",
                "description": "保持原始数据类型"
            },
            "4bit": {
                "name": "4位量化",
                "target_dtype": "float16 (模拟4位)",
                "description": "使用float16模拟4位量化效果"
            },
            "8bit": {
                "name": "8位量化",
                "target_dtype": "int8/uint8",
                "description": "使用8位整数进行量化"
            },
            "fp8_e4m3fn": {
                "name": "FP8量化",
                "target_dtype": "float8_e4m3fn",
                "description": "使用fp8_e4m3fn格式进行量化"
            }
        }
        
        return strategies.get(bits, strategies["none"])
    
    def _analyze_tensors(self, state_dict: Dict) -> Dict:
        """分析模型中的张量类型和数量"""
        float_tensors = 0
        non_float_tensors = 0
        tensor_shapes = {}
        tensor_dtypes = {}
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                # 更新数据类型统计
                dtype_str = str(tensor.dtype)
                tensor_dtypes[dtype_str] = tensor_dtypes.get(dtype_str, 0) + 1
                
                # 更新形状统计（使用前几个维度作为key）
                shape_key = str(tensor.shape[:3]) if len(tensor.shape) >= 3 else str(tensor.shape)
                tensor_shapes[shape_key] = tensor_shapes.get(shape_key, 0) + 1
                
                # 分类统计
                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    float_tensors += 1
                else:
                    non_float_tensors += 1
            else:
                non_float_tensors += 1
        
        # 打印前5种最常见的形状
        print("最常见的张量形状:")
        sorted_shapes = sorted(tensor_shapes.items(), key=lambda x: x[1], reverse=True)[:5]
        for shape, count in sorted_shapes:
            print(f"  - {shape}: {count} 个")
        
        # 打印数据类型分布
        print("张量数据类型分布:")
        for dtype, count in tensor_dtypes.items():
            print(f"  - {dtype}: {count} 个")
        
        return {
            "float_tensors": float_tensors,
            "non_float_tensors": non_float_tensors,
            "tensor_shapes": tensor_shapes,
            "tensor_dtypes": tensor_dtypes
        }
    
    def _quantize_to_8bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """8位量化实现 - 真正的uint8量化以减小模型大小"""
        try:
            # 转换为float32以进行量化
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            
            # 计算量化参数
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            # 避免除零错误
            if min_val == max_val:
                # 如果所有值相同，使用uint8存储常数
                return torch.full_like(tensor, min_val, dtype=torch.uint8)
            
            # 缩放到0-255范围（无符号8位）
            scale = 255.0 / (max_val - min_val)
            zero_point = -min_val * scale
            
            # 应用量化
            quantized = torch.round(tensor * scale + zero_point)
            
            # 裁剪到有效范围
            quantized = torch.clamp(quantized, 0, 255)
            
            # 返回真正的uint8类型张量，这将减小模型大小
            return quantized.to(torch.uint8)
        except Exception as e:
            print(f"8位量化出错: {str(e)}")
            # 出错时返回原始类型
            return tensor.half() if tensor.dtype == torch.float32 else tensor
    
    def _quantize_to_4bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """4位量化实现（使用分组量化策略）"""
        try:
            # 对于4位量化，我们使用float16并应用更激进的缩放
            # 将张量分成多个块，分别量化
            block_size = 16  # 块大小
            
            # 保存原始形状
            original_shape = tensor.shape
            
            # 重塑为更容易处理的形状
            if tensor.dim() >= 2:
                # 对于矩阵或更高维度，保持最后两个维度
                reshaped = tensor.view(-1, original_shape[-2], original_shape[-1])
            else:
                # 对于向量，直接使用
                reshaped = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
            
            # 初始化结果张量
            quantized_blocks = []
            
            # 分组量化
            for block_idx in range(0, reshaped.size(0), block_size):
                end_idx = min(block_idx + block_size, reshaped.size(0))
                block = reshaped[block_idx:end_idx]
                
                # 计算每个块的量化参数
                min_val = block.min()
                max_val = block.max()
                
                # 避免除零错误
                if min_val != max_val:
                    # 缩放到0-15范围（4位）
                    scale = 15.0 / (max_val - min_val)
                    zero_point = -min_val * scale
                    
                    # 应用量化和反量化
                    quantized = torch.round(block * scale + zero_point)
                    quantized = torch.clamp(quantized, 0, 15)
                    dequantized = (quantized - zero_point) / scale
                    
                    quantized_blocks.append(dequantized)
                else:
                    # 如果所有值相同，直接使用
                    quantized_blocks.append(block)
            
            # 合并所有块
            if quantized_blocks:
                result = torch.cat(quantized_blocks, dim=0)
                # 重塑回原始形状
                return result.view(original_shape).half()
            else:
                return tensor.half() if tensor.dtype == torch.float32 else tensor
        except Exception as e:
            print(f"4位量化出错: {str(e)}")
            # 出错时返回float16
            return tensor.half()
    
    def _quantize_to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """FP8量化实现，支持float8_e4m3fn格式"""
        try:
            # 检查是否支持float8
            if hasattr(torch, 'float8_e4m3fn'):
                print("使用原生float8_e4m3fn格式进行量化")
                # 转换为float32以进行量化
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()
                # 转换为float8_e4m3fn格式
                return tensor.to(torch.float8_e4m3fn)
            else:
                # 如果不支持float8，使用更优化的float16作为替代
                print("当前PyTorch版本不支持float8_e4m3fn格式，使用优化的float16代替")
                
                # 对于替代实现，我们使用float16并进行范围优化
                if tensor.dtype == torch.float32:
                    # 分析张量分布
                    mean_val = tensor.mean().item()
                    std_val = tensor.std().item()
                    
                    # 应用轻微的范围压缩，减少精度损失
                    if std_val > 0:
                        # 标准化到更适合float16的范围
                        normalized = (tensor - mean_val) / (std_val * 4.0)  # 缩小范围
                        # 应用轻微的非线性映射以保留更多信息
                        normalized = torch.tanh(normalized) * 0.9  # 平滑边界
                        # 恢复到原始范围的近似值
                        restored = normalized * (std_val * 4.0) + mean_val
                        return restored.half()
                    else:
                        # 如果所有值都相同，直接返回half
                        return tensor.half()
                else:
                    # 已经是float16，直接返回
                    return tensor
        except Exception as e:
            print(f"FP8量化出错: {str(e)}")
            # 出错时返回float16
            return tensor.half() if tensor.dtype == torch.float32 else tensor

    def _merge_lora(self, main_model: str, lora_file: str, output_path: str, scale: float, enable_detailed_logging: bool = False) -> str:
        """将LORA模型融合到主模型中，支持完善的错误处理和内存优化"""
        print("=" * 80)
        print(f"开始执行LoRA融合操作")
        print(f"- 主模型: {main_model}")
        print(f"- LoRA模型: {lora_file}")
        print(f"- 输出路径: {output_path}")
        print(f"- 融合比例: {scale}")
        print(f"- 详细日志: {'已启用' if enable_detailed_logging else '未启用'}")
        print("=" * 80)
        
        # 记录开始时间
        start_time = time.time()
        
        # 加载主模型和LORA模型
        try:
            print("正在加载主模型...")
            main_state_dict = load_file(main_model)
            main_key_count = len(main_state_dict)
            print(f"主模型加载完成，包含 {main_key_count} 个键值对")
            
            if enable_detailed_logging:
                # 输出主模型的一些关键信息
                first_few_keys = list(main_state_dict.keys())[:5]
                print(f"主模型前5个键: {first_few_keys}")
                # 尝试获取第一个权重的形状信息
                if first_few_keys:
                    first_key = first_few_keys[0]
                    first_weight = main_state_dict[first_key]
                    print(f"第一个权重信息 - 键: {first_key}, 形状: {first_weight.shape}, 类型: {type(first_weight).__name__}")
            
            print("正在加载LORA模型...")
            lora_state_dict = load_file(lora_file)
            lora_key_count = len(lora_state_dict)
            print(f"LORA模型加载完成，包含 {lora_key_count} 个键值对")
            
            if enable_detailed_logging:
                # 输出LoRA模型的一些关键信息
                lora_first_few_keys = list(lora_state_dict.keys())[:5]
                print(f"LoRA模型前5个键: {lora_first_few_keys}")
                # 检查是否是标准LoRA格式
                is_standard_lora = any(".lora_down.weight" in k for k in lora_state_dict.keys())
                print(f"是否为标准LoRA格式: {is_standard_lora}")
                
                # 统计不同类型的权重数量
                down_weight_count = sum(".lora_down.weight" in k for k in lora_state_dict.keys())
                up_weight_count = sum(".lora_up.weight" in k for k in lora_state_dict.keys())
                alpha_count = sum(".alpha" in k for k in lora_state_dict.keys())
                print(f"LoRA权重统计: 下投影层 {down_weight_count}, 上投影层 {up_weight_count}, Alpha参数 {alpha_count}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"加载模型失败: {str(e)}")
        
        # 融合LORA权重
        merged_dict = main_state_dict.copy()
        lora_keys = set(lora_state_dict.keys())
        processed_keys = set()
        applied_layers = 0
        total_lora_pairs = 0
        skipped_layers = 0
        
        # 检查是否是标准LORA格式
        has_lora_down = any('.lora_down.' in key for key in lora_keys)
        has_lora_up = any('.lora_up.' in key for key in lora_keys)
        has_loradown = any('.loradown.' in key.lower() for key in lora_keys)
        has_loraup = any('.loraup.' in key.lower() for key in lora_keys)
        has_down_linear = any('.down.linear.' in key for key in lora_keys)
        has_up_linear = any('.up.linear.' in key for key in lora_keys)
        has_lora_A = any('.lora_A.default.weight' in key for key in lora_keys)
        has_lora_B = any('.lora_B.default.weight' in key for key in lora_keys)
        
        # 检查是否找到任何支持的格式
        has_any_format = has_lora_down or has_loradown or has_down_linear or has_lora_A
        
        if not has_any_format:
            print("警告: LORA模型可能不是支持的格式")
            print("支持的格式包括:")
            print("1. .lora_down. / .lora_up.")
            print("2. .loradown. / .loraup. (大小写不敏感)")
            print("3. .down.linear. / .up.linear.")
            print("4. _lora_down. / _lora_up.")
            print("5. down_proj / up_proj")
            print("6. .lora_A.default.weight / .lora_B.default.weight")
            print("\nLoRA模型前10个键示例:")
            for i, key in enumerate(list(lora_keys)[:10]):
                print(f"  {i+1}. {key}")
            print("\n请检查LoRA模型格式是否正确")
        
        # 首先计算总共有多少对lora_down/lora_up权重
        print("正在分析LORA模型结构...")
        # 定义支持的权重对格式
        weight_formats = [
            ('.lora_down.', '.lora_up.'),
            ('_lora_down.', '_lora_up.'),
            ('down_proj', 'up_proj'),
            ('.loradown.', '.loraup.'),  # 小写变体
            ('.down.linear.', '.up.linear.'),
            ('.lora_A.default.weight', '.lora_B.default.weight')  # 新格式
        ]
        
        for key in lora_keys:
            if key in processed_keys:
                continue
            
            # 尝试所有支持的格式
            for down_pattern, up_pattern in weight_formats:
                # 对于大小写不敏感的格式，使用lower()
                if down_pattern in key or ('.loradown.' in down_pattern and down_pattern in key.lower()):
                    # 处理小写变体
                    if '.loradown.' in down_pattern:
                        # 找到小写版本的位置
                        lower_key = key.lower()
                        pos = lower_key.find('.loradown.')
                        if pos != -1:
                            up_key = key[:pos] + '.loraup.' + key[pos + len('.loradown.'):]
                            # 尝试大小写变体
                            possible_up_keys = [up_key, key[:pos] + '.LoraUp.' + key[pos + len('.loradown.'):]]
                            for possible_key in possible_up_keys:
                                if possible_key in lora_keys:
                                    total_lora_pairs += 1
                                    processed_keys.add(key)
                                    processed_keys.add(possible_key)
                                    break
                            continue
                    else:
                        # 标准格式替换
                        up_key = key.replace(down_pattern, up_pattern)
                        if up_key in lora_keys:
                            total_lora_pairs += 1
                            processed_keys.add(key)
                            processed_keys.add(up_key)
                            break
        
        print(f"发现 {total_lora_pairs} 对 LORA 权重需要应用")
        print("注意: 系统将尝试智能映射不同命名的层，但无法保证完全兼容")
        print("支持的映射模式: img_mod → mod/image_mod, img_mlp → mlp/image_mlp, add_v_proj → v_proj")
        
        # 重置已处理键集合
        processed_keys.clear()
        
        # 应用LORA权重
        for key_idx, key in enumerate(lora_keys, 1):
            if key in processed_keys:
                continue
            
            # 尝试多种格式
            # 1. 标准格式 .lora_down. / .lora_up.
            if '.lora_down.' in key:
                base_key = key.split('.lora_down.')[0]
                up_key = key.replace('.lora_down.', '.lora_up.')
                pair_format = "standard"
            # 2. 变体格式 _lora_down. / _lora_up.
            elif '_lora_down.' in key:
                base_key = key.split('_lora_down.')[0]
                up_key = key.replace('_lora_down.', '_lora_up.')
                pair_format = "variant_1"
            # 3. 另一种变体格式 down_proj / up_proj
            elif 'down_proj' in key and 'up_proj' not in key:
                base_key = key.replace('down_proj', '')
                up_key = key.replace('down_proj', 'up_proj')
                pair_format = "variant_2"
            # 4. 小写变体 .loradown. / .loraup.
            elif '.loradown.' in key.lower():
                lower_key = key.lower()
                pos = lower_key.find('.loradown.')
                if pos != -1:
                    base_key = key[:pos]
                    # 尝试多种可能的up键格式
                    possible_up_keys = [
                        key[:pos] + '.loraup.' + key[pos + len('.loradown.'):],
                        key[:pos] + '.LoraUp.' + key[pos + len('.loradown.'):],
                        key[:pos] + '.Lora_up.' + key[pos + len('.loradown.'):]
                    ]
                    up_key = None
                    for possible_key in possible_up_keys:
                        if possible_key in lora_keys:
                            up_key = possible_key
                            break
                    if not up_key:
                        continue
                    pair_format = "variant_lowercase"
                else:
                    continue
            # 5. 线性层格式 .down.linear. / .up.linear.
            elif '.down.linear.' in key:
                base_key = key.split('.down.linear.')[0]
                up_key = key.replace('.down.linear.', '.up.linear.')
                pair_format = "variant_linear"
            # 6. 新格式 .lora_A.default.weight / .lora_B.default.weight
            elif '.lora_A.default.weight' in key:
                # 完全重写的新格式处理逻辑
                # 直接使用完整的键路径作为标识符，但分别提取A/B权重
                base_identifier = key  # 保持原始键用于标识
                # 确保up_key构建正确
                up_key = key.replace('.lora_A.default.weight', '.lora_B.default.weight')
                # 特殊处理：对于新格式，我们直接使用键中的权重，不再尝试提取base_key
                # 这是关键改进 - 我们不再假设base_key模式，而是直接处理键中的实际权重
                pair_format = "variant_new_format"
                # 添加调试信息
                print(f"  处理variant_new_format格式: 原始键={key}")
            else:
                continue
            
            if up_key in lora_keys:
                # 查找层映射，处理主模型中可能不存在完全匹配的层
                # 对于variant_new_format格式，我们使用一个全新的映射策略
                if pair_format == "variant_new_format":
                    # 全新的映射策略：直接查找主模型中与键路径相似的权重
                    print(f"  应用全新的variant_new_format映射策略...")
                    mapped_key = None
                    
                    # 1. 首先尝试直接匹配原始键（不进行任何替换）
                    if key in main_state_dict:
                        mapped_key = key
                        print(f"  - 直接匹配成功: {key}")
                    # 2. 尝试移除.lora_A.default.weight后缀
                    else:
                        # 提取可能的层名（移除.lora_A.default.weight）
                        potential_layer = key.replace('.lora_A.default.weight', '')
                        print(f"  - 尝试潜在层名: {potential_layer}")
                        
                        # 尝试多种可能的权重名称变体
                        potential_keys = [
                            potential_layer,                      # 直接尝试层名
                            f"{potential_layer}.weight",           # 添加.weight
                            f"{potential_layer}.bias",             # 添加.bias
                            potential_layer.replace('.', '/'),     # 替换.为/
                            f"{potential_layer.replace('.', '/')}.weight"  # 组合变体
                        ]
                        
                        # 遍历尝试所有潜在的键
                        for test_key in potential_keys:
                            if test_key in main_state_dict:
                                mapped_key = test_key
                                print(f"  - 找到匹配: {test_key}")
                                break
                else:
                    # 标准格式的处理逻辑
                    mapped_key = base_key
                    print(f"正在处理层: {base_key}")
                    
                    # 检查主模型中是否存在对应的基础层
                    if base_key not in main_state_dict:
                        print(f"  基础层不存在，开始尝试映射策略...")
                        simplified_key = base_key
                    
                    # 全面的层名映射策略 - 增强版
                    original_key = base_key
                    has_weight = base_key.endswith('.weight')
                    base_key_no_weight = base_key[:-7] if has_weight else base_key
                    
                    # 1. 提取块号和子层信息（如果是transformer块结构）
                    block_num = None
                    attention_part = None
                    mlp_part = None
                    
                    parts = base_key_no_weight.split('.')
                    if len(parts) >= 4 and parts[0] == 'transformer_blocks':
                        block_num = parts[1]
                        if len(parts) >= 3 and parts[2] == 'attn':
                            attention_part = parts[2]
                        elif len(parts) >= 3 and parts[2] == 'attention':
                            attention_part = parts[2]
                        elif len(parts) >= 3 and parts[2] == 'mlp':
                            mlp_part = parts[2]
                        print(f"  识别到transformer块: {block_num}, 子层类型: {attention_part or mlp_part}")
                    
                    # 2. 基础映射策略 - 直接替换
                    base_mappings = [
                        ('.img_mod.', '.mod.'),
                        ('.img_mlp.', '.mlp.'),
                        ('.add_', '.'),
                        ('.to_add_out', '.to_out.0'),
                        ('.to_add_out.', '.to_out.0.'),
                        ('.to_out.0', '.to_out'),
                        ('.to_q', '.q_proj'),
                        ('.to_k', '.k_proj'),
                        ('.to_v', '.v_proj'),
                        ('.q_proj', '.to_q'),
                        ('.k_proj', '.to_k'),
                        ('.v_proj', '.to_v'),
                        ('.out_proj', '.to_out.0'),
                        ('.to_out.0', '.out_proj')
                    ]
                    
                    # 尝试基础映射 - 同时测试带weight和不带weight的版本
                    print(f"  尝试基础映射策略...")
                    for pattern, replacement in base_mappings:
                        if pattern in base_key_no_weight:
                            test_key_no_weight = base_key_no_weight.replace(pattern, replacement)
                            test_keys = [test_key_no_weight]  # 不带weight的版本
                            if has_weight:
                                test_keys.append(f"{test_key_no_weight}.weight")  # 带weight的版本
                            
                            for test_key in test_keys:
                                if test_key in main_state_dict:
                                    simplified_key = test_key
                                    print(f"  - 基础映射成功: {original_key} -> {simplified_key} (替换 {pattern} 为 {replacement})")
                                    break
                            if simplified_key != original_key:
                                break
                    
                    # 3. 复杂映射策略 - 针对transformer_blocks中的attn层
                    if simplified_key == original_key and block_num and (attention_part or mlp_part):
                        print(f"  尝试复杂块映射策略...")
                        # 生成所有可能的注意力层变体组合
                        all_variants = []
                        
                        # 子层类型（attn或attention）
                        sublayer_types = ['attn', 'attention'] if attention_part else ['mlp']
                        
                        # 构建变体
                        for sublayer_type in sublayer_types:
                            # 核心注意力层变体
                            if attention_part:
                                # 查询层变体
                                if '.to_q' in base_key_no_weight or '.q_proj' in base_key_no_weight:
                                    core_variants = ['to_q', 'q_proj', 'attn.q', 'attention.to_q']
                                # 键层变体
                                elif '.to_k' in base_key_no_weight or '.k_proj' in base_key_no_weight:
                                    core_variants = ['to_k', 'k_proj', 'attn.k', 'attention.to_k']
                                # 值层变体
                                elif '.to_v' in base_key_no_weight or '.v_proj' in base_key_no_weight:
                                    core_variants = ['to_v', 'v_proj', 'attn.v', 'attention.to_v']
                                # 输出层变体
                                elif '.to_out' in base_key_no_weight or '.out_proj' in base_key_no_weight or '.to_add_out' in base_key_no_weight:
                                    core_variants = ['to_out.0', 'to_out', 'out_proj', 'attn.out', 'attention.to_out.0', 'to_add_out']
                                else:
                                    core_variants = []
                            else:
                                # MLP层变体
                                core_variants = []
                                if '.gate_proj' in base_key_no_weight:
                                    core_variants = ['gate_proj']
                                elif '.down_proj' in base_key_no_weight:
                                    core_variants = ['down_proj']
                                elif '.up_proj' in base_key_no_weight:
                                    core_variants = ['up_proj']
                            
                            # 生成完整变体
                            for core in core_variants:
                                # 处理嵌套结构
                                if '.' in core:
                                    # 对于如'attn.q'这样的嵌套结构
                                    sub_parts = core.split('.')
                                    variant = f'transformer_blocks.{block_num}.{sublayer_type}.{sub_parts[1]}'
                                else:
                                    # 直接结构
                                    variant = f'transformer_blocks.{block_num}.{sublayer_type}.{core}'
                                
                                # 添加到所有变体列表
                                all_variants.append(variant)
                        
                        # 去重
                        all_variants = list(set(all_variants))
                        print(f"  生成了 {len(all_variants)} 个变体进行测试")
                        
                        # 测试变体 - 同时测试带weight和不带weight的版本
                        for variant in all_variants:
                            test_keys = [variant]  # 不带weight的版本
                            if has_weight:
                                test_keys.append(f"{variant}.weight")  # 带weight的版本
                            
                            for test_key in test_keys:
                                if test_key in main_state_dict and test_key != original_key:
                                    simplified_key = test_key
                                    print(f"  - 复杂块映射成功: {original_key} -> {simplified_key} (块{block_num} {sublayer_type}变体)")
                                    break
                            if simplified_key != original_key:
                                break
                    
                    # 4. 尝试完全移除层名的某些部分
                    if simplified_key == original_key:
                        print(f"  尝试移除模式映射策略...")
                        # 尝试移除某些中间部分
                        remove_patterns = ['add_', 'img_', 'image_']
                        for pattern in remove_patterns:
                            if pattern in base_key_no_weight:
                                test_key_no_weight = base_key_no_weight.replace(pattern, '')
                                test_keys = [test_key_no_weight]  # 不带weight的版本
                                if has_weight:
                                    test_keys.append(f"{test_key_no_weight}.weight")  # 带weight的版本
                                
                                for test_key in test_keys:
                                    if test_key in main_state_dict:
                                        simplified_key = test_key
                                        print(f"  - 移除模式映射成功: {original_key} -> {simplified_key} (移除 {pattern})")
                                        break
                                if simplified_key != original_key:
                                    break
                    
                    # 5. 终极映射尝试 - 直接查找相似层名
                    if simplified_key == original_key and block_num:
                        print(f"  尝试相似层映射策略...")
                        # 对于variant_new_format格式，添加特殊的块层映射
                        if pair_format == "variant_new_format":
                            print(f"  - 应用variant_new_format格式的特殊块映射...")
                            # 尝试不同的块层命名变体
                            block_variants = []
                            # 基础块路径
                            base_block = f'transformer_blocks.{block_num}'
                            
                            # 尝试不同的子层命名
                            if attention_part or mlp_part:
                                sublayer = attention_part or mlp_part
                                # 提取层类型（q/k/v/out）
                                layer_type = None
                                if any(x in base_key_no_weight for x in ['q', 'query']):
                                    layer_type = 'q'
                                elif any(x in base_key_no_weight for x in ['k', 'key']):
                                    layer_type = 'k'
                                elif any(x in base_key_no_weight for x in ['v', 'value']):
                                    layer_type = 'v'
                                elif any(x in base_key_no_weight for x in ['out', 'output']):
                                    layer_type = 'out'
                                
                                # 生成常见的层命名变体
                                if layer_type:
                                    if layer_type == 'q':
                                        layer_variants = ['to_q', 'q_proj', 'attn.q', 'attention.to_q']
                                    elif layer_type == 'k':
                                        layer_variants = ['to_k', 'k_proj', 'attn.k', 'attention.to_k']
                                    elif layer_type == 'v':
                                        layer_variants = ['to_v', 'v_proj', 'attn.v', 'attention.to_v']
                                    else:  # out
                                        layer_variants = ['to_out.0', 'out_proj', 'attn.out', 'attention.to_out']
                                    
                                    # 生成完整的块层变体
                                    for layer_var in layer_variants:
                                        variant = f'{base_block}.{sublayer}.{layer_var}'
                                        block_variants.append(variant)
                                        # 也添加带weight的版本
                                        block_variants.append(f'{variant}.weight')
                            
                            # 测试这些变体
                            for variant in block_variants:
                                if variant in main_state_dict and variant != original_key:
                                    simplified_key = variant
                                    print(f"  - 新格式块映射成功: {original_key} -> {simplified_key}")
                                    break
                            
                            if simplified_key != original_key:
                                # 找到匹配，不需要继续其他映射策略
                                pass
                        # 查找相同块中可能相似的层
                        target_prefix = f'transformer_blocks.{block_num}.{attention_part or mlp_part}.to_'
                        found_similar = False
                        for key in main_state_dict:
                            if key.startswith(target_prefix) and key != original_key:
                                # 确保键确实存在于main_state_dict中
                                if key in main_state_dict:
                                    # 检查是否是同一类层（q/k/v/out）
                                    if (('.q' in original_key and '.q' in key) or 
                                        ('.k' in original_key and '.k' in key) or 
                                        ('.v' in original_key and '.v' in key) or 
                                        ('.out' in original_key and '.out' in key)):
                                        simplified_key = key
                                        print(f"  - 相似层映射成功: {original_key} -> {simplified_key} (块{block_num}相似层)")
                                        found_similar = True
                                        break
                        
                        # 如果没找到，尝试更宽松的匹配
                        if not found_similar:
                            print(f"  尝试宽松匹配策略...")
                            # 查找相同块中的所有层
                            block_prefix = f'transformer_blocks.{block_num}.{attention_part or mlp_part}.'
                            
                            # 收集所有可能的候选键并评分
                            candidates = []
                            for key in list(main_state_dict.keys()):
                                if key.startswith(block_prefix) and key != original_key:
                                    # 确保键确实存在
                                    if key in main_state_dict:
                                        score = 0
                                        # 根据关键词匹配度打分
                                        if any(kw in key for kw in ['q', 'k', 'v', 'out', 'proj', 'to_']):
                                            score += 10
                                        # 避免选择bias键，优先选择weight键
                                        if '.bias' not in key:
                                            score += 10  # 增加权重，确保优先选择weight
                                        
                                        # 更高的权重给更相关的层 - 改进版本，更精准的匹配
                                        # 提取原始层类型，包括add_k_proj等特殊情况
                                        orig_layer_type = None
                                        parts = original_key.split('.')
                                        for part in parts:
                                            if part in ['to_q', 'q_proj']:
                                                orig_layer_type = 'q'
                                                break
                                            elif part in ['to_k', 'k_proj']:
                                                orig_layer_type = 'k'
                                                break
                                            elif part in ['to_v', 'v_proj']:
                                                orig_layer_type = 'v'
                                                break
                                            elif part in ['to_out', 'out_proj']:
                                                orig_layer_type = 'out'
                                                break
                                            elif part == 'add_q_proj':
                                                orig_layer_type = 'q'
                                                break
                                            elif part == 'add_k_proj':
                                                orig_layer_type = 'k'
                                                break
                                            elif part == 'add_v_proj':
                                                orig_layer_type = 'v'
                                                break
                                        
                                        # 如果找到原始层类型，给相同类型的目标层更高权重
                                        if orig_layer_type:
                                            # 检查目标键是否包含相同的层类型标识
                                            if ((orig_layer_type == 'q' and ('.q_' in key or '.q.' in key or '.q/' in key or 'add_q_proj' in key)) or 
                                                (orig_layer_type == 'k' and ('.k_' in key or '.k.' in key or '.k/' in key or 'add_k_proj' in key)) or 
                                                (orig_layer_type == 'v' and ('.v_' in key or '.v.' in key or '.v/' in key or 'add_v_proj' in key)) or 
                                                (orig_layer_type == 'out' and ('.out_' in key or '.out.' in key or '.out/' in key))):
                                                score += 30  # 大幅增加同类型层的权重
                                            # 更精确的层名匹配
                                            if f'{orig_layer_type}_proj' in key:
                                                score += 20
                                            # 为add_*_proj类型的键增加额外权重
                                            if 'add_' in key and f'add_{orig_layer_type}_proj' in key:
                                                score += 15  # 为add_类型的键提供额外的分数加成
                                            # 特别为包含weight的键增加额外权重，确保优先选择权重参数
                                            if '.weight' in key:
                                                score += 10
                                        
                                        if score > 0:
                                            candidates.append((score, key))
                            
                            # 按分数排序并选择最高分的候选键
                            if candidates:
                                # 降序排序
                                candidates.sort(reverse=True, key=lambda x: x[0])
                                best_score, best_key = candidates[0]
                                
                                # 如果有多个相同分数的候选，优先选择与原始键类型最匹配的
                                if len(candidates) > 1 and candidates[0][0] == candidates[1][0] and orig_layer_type:
                                    # 首先尝试精确匹配相同的名称格式（add_k_proj匹配add_k_proj）
                                    for score, key in candidates:
                                        if score == best_score:
                                            # 检查是否有完全匹配的层名部分
                                            for part in parts:
                                                if part in key and part in ["add_q_proj", "add_k_proj", "add_v_proj", "q_proj", "k_proj", "v_proj", "out_proj"]:
                                                    best_key = key
                                                    best_score = score
                                                    break
                                            if best_key != candidates[0][1]:
                                                break
                                
                                simplified_key = best_key
                                if enable_detailed_logging:
                                    self.log(f"   - 宽松匹配成功: {original_key} -> {simplified_key} (块{block_num}宽松匹配，匹配度分数: {best_score})")
                    
                    # 如果简化后的键存在，使用它
                    if simplified_key != base_key and simplified_key in main_state_dict:
                        mapped_key = simplified_key
                        if enable_detailed_logging:
                            self.log(f"映射成功: {base_key} -> {mapped_key}")
                    else:
                        if enable_detailed_logging:
                            self.log(f"映射失败: 所有策略都未能找到匹配的层，跳过该LORA权重对（可能是层不兼容）")
                        # 明确设为None，避免使用不存在的键
                        mapped_key = None
                
                # 检查映射后的键是否存在且有效
                # 增强的回退机制，适用于所有格式
                if mapped_key is None:
                    print(f"  尝试增强的回退映射策略...")
                    
                    # 1. 提取原始键的关键部分作为潜在的权重名称
                    key_parts = key.split('.')
                    
                    # 2. 收集可能的权重名称（从不同位置提取）
                    potential_weights = set()
                    for i in range(max(0, len(key_parts) - 4), len(key_parts)):
                        if len(key_parts[i]) > 2:
                            potential_weights.add(key_parts[i])
                    
                    # 3. 尝试常见的权重名称后缀
                    common_suffixes = ['.weight', '.bias']
                    
                    print(f"  - 提取潜在权重名: {list(potential_weights)}")
                    
                    # 4. 智能搜索策略
                    # 4.1 精确匹配关键部分
                    for potential_weight in potential_weights:
                        for main_key in main_state_dict:
                            # 查找精确匹配的部分
                            parts = main_key.split('.')
                            if potential_weight in parts:
                                # 优先选择带有常见后缀的键
                                for suffix in common_suffixes:
                                    if main_key.endswith(suffix):
                                        mapped_key = main_key
                                        print(f"  - 精确部分匹配成功: 找到键 {main_key}")
                                        break
                                if mapped_key:
                                    break
                        if mapped_key:
                            break
                    
                    # 4.2 如果精确匹配失败，尝试模糊匹配
                    if mapped_key is None:
                        print(f"  - 执行全局相似键搜索...")
                        key_parts = key.split('.')
                        # 提取关键部分进行匹配，降低长度要求以捕获更多可能的匹配
                        key_terms = [part for part in key_parts if len(part) > 2]  # 降低到2个字符以上
                        
                        best_match = None
                        best_score = 0
                        
                        for main_key in main_state_dict:
                            score = 0
                            main_parts = main_key.split('.')
                            
                            # 计算匹配分数
                            # 1. 关键术语匹配
                            for term in key_terms:
                                if term in main_key:
                                    score += 2
                                # 检查是否有完全相同的部分
                                if term in main_parts:
                                    score += 3
                            
                            # 2. 层类型匹配（优先考虑类似的层类型）
                            key_has_attn = any(x in key.lower() for x in ['attn', 'attention'])
                            key_has_mlp = any(x in key.lower() for x in ['mlp', 'feed_forward'])
                            key_has_conv = 'conv' in key.lower()
                            
                            if key_has_attn and any(x in main_key.lower() for x in ['attn', 'attention']):
                                score += 5
                            if key_has_mlp and any(x in main_key.lower() for x in ['mlp', 'feed_forward']):
                                score += 5
                            if key_has_conv and 'conv' in main_key.lower():
                                score += 5
                            
                            # 3. 参数类型匹配
                            if '.weight' in main_key:
                                score += 4
                            elif '.bias' in main_key:
                                score += 2
                            
                            # 4. 避免无关层的高分数
                            if 'norm' in main_key and 'norm' not in key:
                                score -= 2
                            
                            if score > best_score:
                                best_score = score
                                best_match = main_key
                        
                        if best_match and best_score > 5:  # 提高阈值以确保质量
                            mapped_key = best_match
                            print(f"  - 全局搜索成功: 最佳匹配 {best_match} (分数: {best_score})")
                
                # 增强的层映射逻辑，特别是针对cross_attn相关的层
                if mapped_key is not None:
                    # 检查直接映射是否存在
                    if mapped_key in main_state_dict:
                        print(f"应用LORA到: {key} (格式: {pair_format})")
                    else:
                        # 智能尝试替代映射，特别是针对cross_attn相关层
                        print(f"  - 主模型中不存在直接映射: {mapped_key}")
                        # 创建智能替代键
                        alternative_mappings = []
                        
                        # 1. 尝试不同的cross_attn命名变体
                        if 'cross_attn' in mapped_key:
                            alternative_mappings.extend([
                                mapped_key.replace('cross_attn', 'attn1'),  # 常见变体
                                mapped_key.replace('cross_attn', 'attn'),
                                mapped_key.replace('cross_attn', 'attn2'),
                                mapped_key.replace('cross_attn', 'context_attn')
                            ])
                        # 2. 尝试不同的attn命名变体
                        elif 'attn' in mapped_key and 'cross_attn' not in mapped_key:
                            alternative_mappings.extend([
                                mapped_key.replace('attn', 'attn1'),
                                mapped_key.replace('attn', 'attn2'),
                                mapped_key.replace('attn1', 'cross_attn'),
                                mapped_key.replace('attn2', 'cross_attn')
                            ])
                        # 3. 尝试不同的块命名变体
                        if 'blocks.' in mapped_key:
                            alternative_mappings.extend([
                                mapped_key.replace('blocks.', 'layers.'),
                                mapped_key.replace('blocks.', 'block.'),
                                mapped_key.replace('transformer.blocks.', 'model.layers.'),
                                mapped_key.replace('transformer_blocks.', 'transformer_layers.')
                            ])
                        
                        # 尝试所有替代映射
                        found_alternative = False
                        for alt_key in alternative_mappings:
                            if alt_key in main_state_dict:
                                print(f"  - 找到替代映射: {alt_key}")
                                mapped_key = alt_key
                                found_alternative = True
                                break
                        
                        if not found_alternative:
                            print(f"  - 未找到任何替代映射")
                            skipped_layers += 1
                            continue
                    
                    try:
                        # 调试信息：显示当前正在处理的完整键映射关系
                        print(f"\n[DEBUG] 处理层映射: {key} -> {mapped_key}")
                        print(f"[DEBUG] 格式类型: {pair_format}, 缩放因子: {scale}")
                        
                        # 获取权重 - 重写的权重提取逻辑
                        # 对于variant_new_format格式，我们直接使用键中的权重数据，不再区分down/up
                        if pair_format == "variant_new_format":
                            print(f"  - 处理variant_new_format权重...")
                            # 直接获取A和B权重
                            if key not in lora_state_dict:
                                print(f"  - 错误: LORA A权重 {key} 不存在")
                                skipped_layers += 1
                                continue
                            if up_key not in lora_state_dict:
                                print(f"  - 错误: LORA B权重 {up_key} 不存在")
                                skipped_layers += 1
                                continue
                                 
                            lora_a = lora_state_dict[key]
                            lora_b = lora_state_dict[up_key]
                            print(f"  - 成功获取A/B权重，形状: A={lora_a.shape}, B={lora_b.shape}")
                            # 添加权重统计信息
                            print(f"  - LORA A权重统计: 均值={lora_a.mean():.6f}, 标准差={lora_a.std():.6f}, 最小值={lora_a.min():.6f}, 最大值={lora_a.max():.6f}")
                            print(f"  - LORA B权重统计: 均值={lora_b.mean():.6f}, 标准差={lora_b.std():.6f}, 最小值={lora_b.min():.6f}, 最大值={lora_b.max():.6f}")
                        else:
                            # 标准格式的权重提取
                            if key not in lora_state_dict:
                                print(f"  - 警告: LORA down键 {key} 不存在")
                                skipped_layers += 1
                                continue
                            if up_key not in lora_state_dict:
                                print(f"  - 警告: LORA up键 {up_key} 不存在")
                                skipped_layers += 1
                                continue
                                 
                            lora_down = lora_state_dict[key]
                            lora_up = lora_state_dict[up_key]
                            # 添加权重统计信息
                            print(f"  - LORA down权重统计: 均值={lora_down.mean():.6f}, 标准差={lora_down.std():.6f}, 最小值={lora_down.min():.6f}, 最大值={lora_down.max():.6f}")
                            print(f"  - LORA up权重统计: 均值={lora_up.mean():.6f}, 标准差={lora_up.std():.6f}, 最小值={lora_up.min():.6f}, 最大值={lora_up.max():.6f}")
                        
                        # 记录权重信息
                        if pair_format == "variant_new_format":
                            print(f"  - LORA A形状: {lora_a.shape}, 类型: {lora_a.dtype}, 元素数量: {lora_a.numel()}")
                            print(f"  - LORA B形状: {lora_b.shape}, 类型: {lora_b.dtype}, 元素数量: {lora_b.numel()}")
                             
                            # 确保数据类型一致性
                            if lora_a.dtype != lora_b.dtype:
                                print(f"  - 警告: LORA A和B权重数据类型不一致，进行转换")
                                if lora_a.dtype == torch.float32:
                                    lora_b = lora_b.float()
                                else:
                                    lora_a = lora_a.to(lora_b.dtype)
                                print(f"  - 转换后: A={lora_a.dtype}, B={lora_b.dtype}")
                        else:
                            print(f"  - LORA down形状: {lora_down.shape}, 类型: {lora_down.dtype}, 元素数量: {lora_down.numel()}")
                            print(f"  - LORA up形状: {lora_up.shape}, 类型: {lora_up.dtype}, 元素数量: {lora_up.numel()}")
                             
                            # 确保数据类型一致性
                            if lora_down.dtype != lora_up.dtype:
                                print(f"  - 警告: LORA down和up权重数据类型不一致，进行转换")
                                if lora_down.dtype == torch.float32:
                                    lora_up = lora_up.float()
                                else:
                                    lora_down = lora_down.to(lora_up.dtype)
                                print(f"  - 转换后: down={lora_down.dtype}, up={lora_up.dtype}")
                        
                        # 计算delta权重
                        try:
                            # 调试信息：显示当前正在处理的键和权重兼容性分析
                            print(f"  - 正在计算delta...")
                            
                            # 添加详细的矩阵乘法兼容性分析
                            if pair_format == "variant_new_format":
                                print(f"  - 应用variant_new_format格式的专用乘法逻辑...")
                                # 确保权重存在且有效
                                if lora_a is None or lora_b is None:
                                    print(f"  - 错误: LORA权重为None")
                                    skipped_layers += 1
                                    continue
                                 
                                # 矩阵乘法兼容性预检查
                                print(f"  - 矩阵乘法兼容性分析: A.shape[-1]={lora_a.shape[-1]}, B.shape[0]={lora_b.shape[0]}")
                                print(f"  - A转置后兼容性: A.T.shape[-1]={lora_a.T.shape[-1]}, B.shape[0]={lora_b.shape[0]}")
                                
                                # 对于新格式，我们尝试多种可能的乘法组合
                                delta = None
                                # 重新排序乘法尝试，优先最常见的组合（LoRA标准是先B后A转置）
                                multiplication_attempts = [
                                    ("标准LoRA顺序 (B × A^T)", lambda: torch.matmul(lora_b, lora_a.T)),
                                    ("默认顺序", lambda: torch.matmul(lora_b, lora_a)),
                                    ("A转置", lambda: torch.matmul(lora_b, lora_a.T)),
                                    ("B转置", lambda: torch.matmul(lora_b.T, lora_a)),
                                    ("反向顺序", lambda: torch.matmul(lora_a, lora_b)),
                                    ("双转置", lambda: torch.matmul(lora_b.T, lora_a.T)),
                                    ("结果转置", lambda: torch.matmul(lora_b, lora_a).T),
                                    ("反向结果转置", lambda: torch.matmul(lora_a, lora_b).T)
                                ]
                                
                                for attempt_name, multiply_func in multiplication_attempts:
                                    try:
                                        delta = multiply_func() * scale
                                        print(f"  - 乘法成功: {attempt_name}，形状: {delta.shape}")
                                        # 添加delta权重统计信息
                                        print(f"  - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}, 最小值={delta.min():.6f}, 最大值={delta.max():.6f}")
                                        break
                                    except RuntimeError as e:
                                        # 更详细的错误信息
                                        error_msg = str(e)
                                        print(f"  - 乘法尝试失败 {attempt_name}: {error_msg}")
                                        # 尝试捕获形状错误并提供更具体的诊断
                                        if 'shape' in error_msg.lower():
                                            print(f"    - lora_a形状: {lora_a.shape}, lora_b形状: {lora_b.shape}")
                                            print(f"    - 矩阵维度不匹配分析: 期望维度 compatible with {lora_b.shape} × {lora_a.shape}")
                                            # 尝试显式处理维度不匹配
                                            try:
                                                # 尝试调整维度以匹配
                                                if lora_a.ndim == 3 and lora_b.ndim == 2:
                                                    # 从3D降到2D
                                                    lora_a_2d = lora_a.view(-1, lora_a.shape[-1])
                                                    print(f"    - 尝试将lora_a从3D调整为2D: {lora_a_2d.shape}")
                                                    delta = torch.matmul(lora_b, lora_a_2d) * scale
                                                    print(f"    - 调整后乘法成功，形状: {delta.shape}")
                                                    # 添加delta权重统计信息
                                                    print(f"    - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}")
                                                    break
                                                elif lora_a.ndim == 2 and lora_b.ndim == 3:
                                                    # 从3D降到2D
                                                    lora_b_2d = lora_b.view(-1, lora_b.shape[-1])
                                                    print(f"    - 尝试将lora_b从3D调整为2D: {lora_b_2d.shape}")
                                                    delta = torch.matmul(lora_b_2d, lora_a) * scale
                                                    print(f"    - 调整后乘法成功，形状: {delta.shape}")
                                                    # 添加delta权重统计信息
                                                    print(f"    - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}")
                                                    break
                                                # 添加3D到3D的调整尝试
                                                elif lora_a.ndim == 3 and lora_b.ndim == 3:
                                                    print(f"    - 尝试3D张量特殊处理...")
                                                    # 尝试保持最后一维，合并前两维
                                                    lora_a_reshaped = lora_a.view(-1, lora_a.shape[-1])
                                                    lora_b_reshaped = lora_b.view(-1, lora_b.shape[-1])
                                                    print(f"    - 重塑后: A={lora_a_reshaped.shape}, B={lora_b_reshaped.shape}")
                                                    try:
                                                        delta = torch.matmul(lora_b_reshaped, lora_a_reshaped.T) * scale
                                                        print(f"    - 3D张量重塑后乘法成功: {delta.shape}")
                                                        # 添加delta权重统计信息
                                                        print(f"    - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}")
                                                        break
                                                    except Exception as inner_e2:
                                                        print(f"    - 3D张量重塑尝试失败: {str(inner_e2)}")
                                            except Exception as inner_e:
                                                print(f"    - 维度调整尝试失败: {str(inner_e)}")
                                
                                if delta is None:
                                    print(f"  - 所有乘法尝试均失败")
                                    skipped_layers += 1
                                    continue
                            else:
                                # 确保权重存在且有效
                                if lora_down is None or lora_up is None:
                                    print(f"  - 错误: LORA权重为None")
                                    skipped_layers += 1
                                    continue
                                 
                                # 标准格式的矩阵乘法预检查
                                print(f"  - 标准格式矩阵乘法分析: up.shape={lora_up.shape}, down.shape={lora_down.shape}")
                                print(f"  - 兼容性检查: up.shape[-1]={lora_up.shape[-1]} vs down.shape[0]={lora_down.shape[0]}")
                                
                                # 标准格式的矩阵乘法
                                delta = torch.matmul(lora_up, lora_down) * scale
                                print(f"  - 计算delta完成，形状: {delta.shape}")
                                # 添加delta权重统计信息
                                print(f"  - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}, 最小值={delta.min():.6f}, 最大值={delta.max():.6f}")
                        except RuntimeError as e:
                            print(f"  - 矩阵乘法失败: {str(e)}")
                            # 根据不同的格式使用正确的变量名称
                            if pair_format == "variant_new_format":
                                print(f"  - lora_a维度: {lora_a.shape}, lora_b维度: {lora_b.shape}")
                                print(f"  - 维度不匹配详细分析: 无法执行 {lora_b.shape} × {lora_a.shape} 或其变体")
                                # 尝试多种乘法顺序 - 已经在variant_new_format专用逻辑中尝试过
                                print(f"  - 所有乘法尝试均失败，跳过此层")
                                skipped_layers += 1
                                continue
                            else:
                                print(f"  - lora_up维度: {lora_up.shape}, lora_down维度: {lora_down.shape}")
                                print(f"  - 维度不匹配详细分析: 标准乘法要求 up.shape[-1] == down.shape[0]")
                                # 尝试多种乘法顺序
                                try:
                                    # 尝试转置乘法
                                    print(f"  - 尝试转置乘法策略: down × up 然后转置")
                                    delta = torch.matmul(lora_down, lora_up).T * scale
                                    print(f"  - 尝试转置乘法成功，形状: {delta.shape}")
                                    # 添加delta权重统计信息
                                    print(f"  - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}")
                                except RuntimeError as e2:
                                    try:
                                        # 尝试另一种转置组合
                                        print(f"  - 尝试另一种转置组合: up.T × down 然后转置")
                                        delta = torch.matmul(lora_up.T, lora_down).T * scale
                                        print(f"  - 尝试另一种转置组合成功，形状: {delta.shape}")
                                        # 添加delta权重统计信息
                                        print(f"  - delta权重统计: 均值={delta.mean():.6f}, 标准差={delta.std():.6f}")
                                    except RuntimeError as e3:
                                        print(f"  - 所有乘法尝试均失败，跳过此层")
                                        skipped_layers += 1
                                        continue
                        
                        # 应用delta到主模型权重 - 完全重写的错误处理逻辑
                        # 1. 检查映射键是否存在
                        if mapped_key not in main_state_dict:
                            print(f"  - 警告: 主模型中不存在映射后的键 {mapped_key}")
                            # 调试信息：显示主模型中类似前缀的键
                            similar_keys = []
                            prefix_parts = mapped_key.split('.')[:2]  # 获取前两部分作为前缀
                            prefix = '.'.join(prefix_parts)
                            for main_key in main_state_dict:
                                if main_key.startswith(prefix):
                                    similar_keys.append(main_key)
                                    if len(similar_keys) > 5:  # 限制显示数量
                                        similar_keys.append("...")
                                        break
                            print(f"  - 主模型中相似前缀的键: {similar_keys}")
                            
                            # 尝试查找相似的键作为备选
                            alternative_key = None
                            for main_key in main_state_dict:
                                # 查找相同块的相似层
                                if mapped_key.split('.')[0] == main_key.split('.')[0] and mapped_key.split('.')[1] == main_key.split('.')[1]:
                                    # 检查是否是相似类型的层（注意力层或MLP层）
                                    mapped_parts = mapped_key.split('.')
                                    main_parts = main_key.split('.')
                                    if len(mapped_parts) >= 3 and len(main_parts) >= 3:
                                        # 比较子层类型（attn/mlp）
                                        if mapped_parts[2] == main_parts[2]:
                                            alternative_key = main_key
                                            break
                             
                            if alternative_key:
                                print(f"  - 找到备选键: {alternative_key}")
                                mapped_key = alternative_key
                            else:
                                print(f"  - 未找到备选键，跳过此层")
                                skipped_layers += 1
                                continue
                        
                        # 2. 安全地应用delta
                        try:
                            main_weight = main_state_dict[mapped_key]
                            main_shape = main_weight.shape
                            
                            # 添加主模型权重统计信息
                            print(f"  - 主模型权重信息: 形状={main_shape}, 类型={main_weight.dtype}, 元素数量={main_weight.numel()}")
                            print(f"  - 主模型权重统计: 均值={main_weight.mean():.6f}, 标准差={main_weight.std():.6f}")
                             
                            # 3. 形状兼容性检查
                            if main_shape == delta.shape:
                                print(f"  - 形状完全匹配! 主模型 {main_shape} == delta {delta.shape}")
                                # 强制将delta转换为主模型的dtype，确保类型一致性
                                print(f"  - 类型转换: delta {delta.dtype} -> {main_weight.dtype}")
                                delta = delta.to(main_weight.dtype)
                                merged_dict[mapped_key] = main_weight + delta
                                applied_layers += 1
                                 
                                # 记录进度
                                progress = (applied_layers / total_lora_pairs) * 100
                                memory_usage = self._get_memory_usage()
                                print(f"LORA融合进度: {progress:.1f}% - 已应用 {applied_layers}/{total_lora_pairs} 层 - 内存使用: {memory_usage:.2f} MB")
                            else:
                                # 增强的智能形状适配逻辑，适用于所有格式
                                print(f"  - 形状不匹配: 主模型 {main_shape} vs delta {delta.shape}")
                                print(f"  - 形状差异分析: 维度数={len(main_shape)} vs {len(delta.shape)}, 元素数量={main_weight.numel()} vs {delta.numel()}")
                                shape_matched = False
                                    
                                # 1. 尝试基本转置匹配
                                if main_shape[::-1] == delta.shape:
                                    print(f"  - 转置匹配成功")
                                    delta = delta.T
                                    shape_matched = True
                                # 优先尝试常见的LoRA形状修复：如果delta是方阵，尝试转置
                                elif len(main_shape) == 2 and len(delta.shape) == 2 and main_shape[0] == delta.shape[1] and main_shape[1] == delta.shape[0]:
                                    print(f"  - 发现可能的LoRA权重转置需求，尝试转置")
                                    delta = delta.T
                                    shape_matched = True
                                # 2. 维度转换策略 - 增强版
                                elif len(main_shape) != len(delta.shape):
                                    # 从2D转换到4D（卷积层）- 增强版
                                    if len(main_shape) == 4 and len(delta.shape) == 2:
                                        print(f"  - 尝试从2D扩展到4D（卷积层模式）...")
                                        try:
                                            # 处理不同的卷积层格式
                                            if main_shape[2:] == (1, 1):  # 1x1卷积
                                                # 尝试两种可能的通道顺序
                                                if delta.shape[0] == main_shape[0] and delta.shape[1] == main_shape[1]:
                                                    delta = delta.view(out_ch, in_ch, 1, 1)
                                                    print(f"  - 1x1卷积维度扩展成功(格式1): {delta.shape}")
                                                    shape_matched = True
                                                elif delta.shape[1] == main_shape[0] and delta.shape[0] == main_shape[1]:
                                                    # 尝试交换通道顺序
                                                    delta = delta.T.view(main_shape[0], main_shape[1], 1, 1)
                                                    print(f"  - 1x1卷积维度扩展成功(格式2): {delta.shape}")
                                                    shape_matched = True
                                            else:
                                                # 非1x1卷积的多种格式尝试
                                                print(f"  - 尝试替代卷积层格式...")
                                                # 尝试直接重塑
                                                try:
                                                    delta = delta.view(main_shape)
                                                    print(f"  - 直接视图调整成功: {delta.shape}")
                                                    shape_matched = True
                                                except:
                                                    # 尝试转置后重塑
                                                    try:
                                                        delta = delta.T.view(main_shape)
                                                        print(f"  - 转置后视图调整成功: {delta.shape}")
                                                        shape_matched = True
                                                    except:
                                                        print(f"  - 标准视图调整失败")
                                        except Exception as e:
                                            print(f"  - 维度扩展失败: {str(e)}")
                                    # 从4D转换到2D - 增强版
                                    elif len(main_shape) == 2 and len(delta.shape) == 4:
                                        print(f"  - 尝试从4D降维到2D...")
                                        try:
                                            # 对于1x1卷积核
                                            if delta.shape[2:] == (1, 1):
                                                # 尝试直接降维
                                                delta = delta.view(delta.shape[0], delta.shape[1])
                                                print(f"  - 4D到2D降维成功(格式1): {delta.shape}")
                                                shape_matched = True
                                            else:
                                                # 对于非1x1卷积，尝试多种降维方式
                                                # 方式1: 直接合并所有维度到2D
                                                try:
                                                    delta_reshaped = delta.view(delta.shape[0] * delta.shape[2] * delta.shape[3], delta.shape[1])
                                                    if delta_reshaped.shape == main_shape:
                                                        delta = delta_reshaped
                                                        print(f"  - 4D到2D降维成功(格式2): {delta.shape}")
                                                        shape_matched = True
                                                except:
                                                    pass
                                                # 方式2: 合并空间维度
                                                try:
                                                    spatial_size = delta.shape[2] * delta.shape[3]
                                                    delta_reshaped = delta.view(delta.shape[0], delta.shape[1], spatial_size).transpose(1, 2).reshape(-1, delta.shape[1])
                                                    if delta_reshaped.shape == main_shape:
                                                        delta = delta_reshaped
                                                        print(f"  - 4D到2D降维成功(格式3): {delta.shape}")
                                                        shape_matched = True
                                                except:
                                                    pass
                                        except Exception as e:
                                            print(f"  - 降维失败: {str(e)}")
                                    # 3D和2D之间的转换 - 增强版
                                    elif len(main_shape) == 3 and len(delta.shape) == 2:
                                        print(f"  - 尝试从2D扩展到3D...")
                                        try:
                                            # 尝试多种常见的扩展模式
                                            # 模式1: 扩展第一维
                                            if main_shape[1:] == delta.shape:
                                                delta = delta.unsqueeze(0).expand_as(main_shape)
                                                print(f"  - 2D到3D扩展成功(模式1): {delta.shape}")
                                                shape_matched = True
                                            # 模式2: 扩展最后一维
                                            elif main_shape[:2] == delta.shape:
                                                delta = delta.unsqueeze(-1).expand_as(main_shape)
                                                print(f"  - 2D到3D扩展成功(模式2): {delta.shape}")
                                                shape_matched = True
                                            # 模式3: 尝试其他维度扩展
                                            elif main_shape[0] * main_shape[1] == delta.shape[0] and main_shape[2] == delta.shape[1]:
                                                delta = delta.reshape(main_shape)
                                                print(f"  - 2D到3D扩展成功(模式3): {delta.shape}")
                                                shape_matched = True
                                        except Exception as e:
                                            print(f"  - 扩展失败: {str(e)}")
                                    elif len(main_shape) == 2 and len(delta.shape) == 3:
                                        print(f"  - 尝试从3D降维到2D...")
                                        try:
                                            # 尝试多种合并维度方式
                                            # 方式1: 合并前两维
                                            delta_reshaped = delta.view(-1, delta.shape[-1])
                                            if delta_reshaped.shape == main_shape:
                                                delta = delta_reshaped
                                                print(f"  - 3D到2D降维成功(方式1): {delta.shape}")
                                                shape_matched = True
                                            # 方式2: 合并后两维
                                            else:
                                                delta_reshaped = delta.reshape(delta.shape[0], -1)
                                                if delta_reshaped.shape == main_shape:
                                                    delta = delta_reshaped
                                                    print(f"  - 3D到2D降维成功(方式2): {delta.shape}")
                                                    shape_matched = True
                                            # 方式3: 完全展平后重塑
                                            if not shape_matched:
                                                total_elements = torch.prod(torch.tensor(delta.shape))
                                                if total_elements == main_shape[0] * main_shape[1]:
                                                    delta = delta.view(main_shape)
                                                    print(f"  - 3D到2D降维成功(方式3): {delta.shape}")
                                                    shape_matched = True
                                        except Exception as e:
                                            print(f"  - 降维失败: {str(e)}")
                                    # 2D和1D之间的转换 - 新增处理逻辑
                                    elif len(main_shape) == 1 and len(delta.shape) == 2:
                                        print(f"  - 尝试从2D降维到1D...")
                                        try:
                                            # 检查是否是偏置项
                                            if '.bias' in mapped_key:
                                                # 对于偏置项，我们可以尝试对delta的每一行进行平均或求和
                                                # 方案1: 如果delta的最后一维与主模型形状匹配，对行进行求和
                                                if delta.shape[1] == main_shape[0]:
                                                    print(f"  - 检测到偏置项，尝试对delta行进行求和")
                                                    delta = delta.sum(dim=0)
                                                    print(f"  - 2D到1D降维成功(求和): {delta.shape}")
                                                    shape_matched = True
                                                # 方案2: 如果delta的第一维与主模型形状匹配，对列进行求和
                                                elif delta.shape[0] == main_shape[0]:
                                                    print(f"  - 检测到偏置项，尝试对delta列进行求和")
                                                    delta = delta.sum(dim=1)
                                                    print(f"  - 2D到1D降维成功(求和): {delta.shape}")
                                                    shape_matched = True
                                                # 方案3: 尝试行平均
                                                elif not shape_matched:
                                                    print(f"  - 尝试对delta行进行平均")
                                                    delta = delta.mean(dim=0)
                                                    print(f"  - 2D到1D降维成功(平均): {delta.shape}")
                                                    shape_matched = True
                                            else:
                                                # 非偏置项，尝试其他降维方式
                                                if delta.shape[1] == main_shape[0]:
                                                    # 选择第一行
                                                    delta = delta[0]
                                                    print(f"  - 2D到1D降维成功(选择第一行): {delta.shape}")
                                                    shape_matched = True
                                                elif delta.shape[0] == main_shape[0]:
                                                    # 选择第一列
                                                    delta = delta[:, 0]
                                                    print(f"  - 2D到1D降维成功(选择第一列): {delta.shape}")
                                                    shape_matched = True
                                        except Exception as e:
                                            print(f"  - 2D到1D降维失败: {str(e)}")
                                
                                # 3. 高级转置组合尝试 - 如果前面的方法未成功
                                if not shape_matched and main_shape != delta.shape:
                                    print(f"  - 尝试多种转置组合...")
                                    # 创建更全面的转置尝试队列
                                    transpose_attempts = []
                                    
                                    # 根据维度数生成不同的转置策略
                                    if len(delta.shape) == 2:
                                        # 2D矩阵转置
                                        transpose_attempts.append(delta.T)  # 标准转置
                                    elif len(delta.shape) == 3:
                                        # 3D矩阵的所有可能转置
                                        transpose_attempts.append(delta.transpose(0, 1))
                                        transpose_attempts.append(delta.transpose(1, 2))
                                        transpose_attempts.append(delta.transpose(0, 2))
                                        # 尝试连续转置
                                        transpose_attempts.append(delta.transpose(0, 1).transpose(1, 2))
                                        # 增加循环转置变体
                                        transpose_attempts.append(delta.permute(1, 2, 0))
                                        transpose_attempts.append(delta.permute(2, 0, 1))
                                    elif len(delta.shape) == 4:
                                        # 4D矩阵的更多转置模式
                                        # 通道相关转置
                                        transpose_attempts.append(delta.transpose(0, 1))  # 交换输入/输出通道
                                        # 空间相关转置
                                        transpose_attempts.append(delta.transpose(2, 3))  # 交换空间维度
                                        # 组合转置
                                        transpose_attempts.append(delta.permute(1, 0, 3, 2))  # 通道和空间双重转置
                                        transpose_attempts.append(delta.permute(0, 1, 3, 2))  # 仅空间维度转置
                                        transpose_attempts.append(delta.permute(1, 0, 2, 3))  # 仅通道转置
                                        # 更复杂的排列
                                        transpose_attempts.append(delta.permute(2, 3, 0, 1))  # 空间维度前置
                                        transpose_attempts.append(delta.permute(3, 2, 1, 0))  # 完全反转所有维度
                                    
                                    # 尝试所有转置方案
                                    for attempt in transpose_attempts:
                                        if attempt.shape == main_shape:
                                            delta = attempt
                                            print(f"  - 转置尝试成功! 新形状: {delta.shape}")
                                            shape_matched = True
                                            break
                                    
                                    # 4. 多维度降维再转置策略 - 增强版
                                    if not shape_matched and len(delta.shape) > 2:
                                        print(f"  - 尝试多种降维转置组合...")
                                        
                                        # 针对3D张量的高级降维策略
                                        if len(delta.shape) == 3:
                                            try:
                                                # 尝试合并不同维度组合
                                                # 组合1: 合并前两维
                                                merged_delta1 = delta.view(-1, delta.shape[2])
                                                if merged_delta1.shape == main_shape:
                                                    delta = merged_delta1
                                                    print(f"  - 降维成功(组合1): {delta.shape}")
                                                    shape_matched = True
                                                elif merged_delta1.T.shape == main_shape:
                                                    delta = merged_delta1.T
                                                    print(f"  - 降维转置成功(组合1): {delta.shape}")
                                                    shape_matched = True
                                                # 组合2: 合并后两维
                                                elif not shape_matched:
                                                    merged_delta2 = delta.reshape(delta.shape[0], -1)
                                                    if merged_delta2.shape == main_shape:
                                                        delta = merged_delta2
                                                        print(f"  - 降维成功(组合2): {delta.shape}")
                                                        shape_matched = True
                                                    elif merged_delta2.T.shape == main_shape:
                                                        delta = merged_delta2.T
                                                        print(f"  - 降维转置成功(组合2): {delta.shape}")
                                                        shape_matched = True
                                            except Exception as e:
                                                print(f"  - 3D降维失败: {str(e)}")
                                        
                                        # 针对4D张量的高级降维策略
                                        elif len(delta.shape) == 4:
                                            try:
                                                # 多种降维组合
                                                # 组合1: 合并通道和空间维度
                                                merged_delta = delta.view(-1, delta.shape[3])
                                                if merged_delta.shape == main_shape:
                                                    delta = merged_delta
                                                    print(f"  - 4D降维成功(组合1): {delta.shape}")
                                                    shape_matched = True
                                                elif merged_delta.T.shape == main_shape:
                                                    delta = merged_delta.T
                                                    print(f"  - 4D降维转置成功(组合1): {delta.shape}")
                                                    shape_matched = True
                                                # 组合2: 仅合并空间维度
                                                elif not shape_matched:
                                                    spatial_size = delta.shape[2] * delta.shape[3]
                                                    merged_delta = delta.reshape(delta.shape[0], delta.shape[1], spatial_size)
                                                    # 尝试不同的排列方式
                                                    for permute_order in [(0, 2, 1), (1, 2, 0), (2, 0, 1)]:
                                                        permuted_delta = merged_delta.permute(permute_order).reshape(-1, merged_delta.shape[2])
                                                        if permuted_delta.shape == main_shape:
                                                            delta = permuted_delta
                                                            print(f"  - 4D降维成功(组合2): {delta.shape}")
                                                            shape_matched = True
                                                            break
                                                        elif hasattr(permuted_delta, 'T') and permuted_delta.T.shape == main_shape:
                                                            delta = permuted_delta.T
                                                            print(f"  - 4D降维转置成功(组合2): {delta.shape}")
                                                            shape_matched = True
                                                            break
                                                    if shape_matched:
                                                        break
                                            except Exception as e:
                                                print(f"  - 4D降维失败: {str(e)}")
                                    
                                # 5. 最终形状检查和应用
                                if shape_matched or main_shape == delta.shape:
                                    print(f"  - 形状适配成功，应用权重更新")
                                    delta = delta.to(main_weight.dtype)
                                    merged_dict[mapped_key] = main_weight + delta
                                    applied_layers += 1
                                else:
                                    # 处理2D矩阵之间的行数不匹配情况
                                    if len(main_shape) == 2 and len(delta.shape) == 2 and main_shape[1] == delta.shape[1]:
                                        print(f"  - 检测到2D矩阵列数匹配但行数不匹配，尝试行裁剪或扩展...")
                                        try:
                                            if delta.shape[0] > main_shape[0]:
                                                # 如果delta行数多于主模型，裁剪delta的行
                                                print(f"  - delta行数({delta.shape[0]})多于主模型({main_shape[0]})，尝试裁剪行")
                                                # 方案1: 取前N行（N=主模型行数）
                                                delta_cropped = delta[:main_shape[0], :]
                                                print(f"  - 裁剪后delta形状: {delta_cropped.shape}")
                                                delta = delta_cropped
                                                shape_matched = True
                                            elif delta.shape[0] < main_shape[0]:
                                                # 如果delta行数少于主模型，尝试扩展delta
                                                print(f"  - delta行数({delta.shape[0]})少于主模型({main_shape[0]})，尝试扩展行")
                                                # 创建与主模型相同形状的零张量
                                                delta_extended = torch.zeros_like(main_weight, dtype=delta.dtype, device=delta.device)
                                                # 将delta填充到前几行
                                                delta_extended[:delta.shape[0], :] = delta
                                                print(f"  - 扩展后delta形状: {delta_extended.shape}")
                                                delta = delta_extended
                                                shape_matched = True
                                            
                                            if shape_matched:
                                                print(f"  - 2D矩阵行适配成功，应用权重更新")
                                                delta = delta.to(main_weight.dtype)
                                                merged_dict[mapped_key] = main_weight + delta
                                                applied_layers += 1
                                                continue
                                        except Exception as e:
                                            print(f"  - 2D矩阵行适配失败: {str(e)}")
                                    
                                    # 最后尝试：如果元素数量匹配但形状不同，强制重塑
                                    try:
                                        if torch.prod(torch.tensor(main_shape)) == torch.prod(torch.tensor(delta.shape)):
                                            print(f"  - 元素数量匹配，尝试强制重塑...")
                                            delta = delta.view(main_shape)
                                            print(f"  - 强制重塑成功! 新形状: {delta.shape}")
                                            delta = delta.to(main_weight.dtype)
                                            merged_dict[mapped_key] = main_weight + delta
                                            applied_layers += 1
                                        else:
                                            print(f"  - 形状适配失败，元素数量也不匹配，跳过此层")
                                            skipped_layers += 1
                                    except Exception as e:
                                        print(f"  - 强制重塑失败: {str(e)}")
                                        skipped_layers += 1
                        except Exception as e:
                            # 捕获所有异常，包括可能的'txt_norm.weight'等特殊键错误
                            error_type = type(e).__name__
                            print(f"  - 应用权重时发生{error_type}: {str(e)}")
                            print(f"  - 详细信息: 处理键 '{mapped_key}' 时出错")
                            print(f"  - 异常上下文: 主模型形状={main_shape if 'main_shape' in locals() else '未知'}, delta形状={delta.shape if 'delta' in locals() else '未知'}")
                            print(f"  - 内存使用情况: {self._get_memory_usage():.2f} MB")
                            skipped_layers += 1
                            
                            # 对于variant_new_format格式，添加增强的最后补救措施
                            if pair_format == "variant_new_format":
                                print(f"  - 应用variant_new_format格式的增强最后补救措施...")
                                
                                try:
                                    # 1. 提取键中的关键字段用于全局搜索
                                    key_parts = key.split('.')
                                    meaningful_parts = [part for part in key_parts if len(part) > 3]
                                    print(f"  - 提取关键字段: {meaningful_parts}")
                                    
                                    # 2. 生成智能替代键
                                    alternative_keys = set()
                                    
                                    # 添加基础变体
                                    base_variants = [
                                        mapped_key.replace('.weight', ''),
                                        mapped_key + '.weight',
                                        mapped_key.replace('transformer_blocks', 'transformer_layers'),
                                        mapped_key.replace('attn', 'attention')
                                    ]
                                    
                                    for base_var in base_variants:
                                        alternative_keys.add(base_var)
                                        # 添加更多变体
                                        if '.weight' not in base_var:
                                            alternative_keys.add(base_var + '.weight')
                                        if '.bias' not in base_var:
                                            alternative_keys.add(base_var + '.bias')
                                    
                                    # 3. 智能搜索类似层
                                    if meaningful_parts:
                                        print(f"  - 执行智能层搜索...")
                                        
                                        # 存储所有可能匹配的候选
                                        candidates = []
                                        
                                        for main_key in main_state_dict:
                                            # 计算匹配分数
                                            score = 0
                                            # 基于关键字匹配
                                            for term in meaningful_parts:
                                                if term in main_key:
                                                    score += 1
                                            # 优先选择weight键
                                            if '.weight' in main_key:
                                                score += 2
                                            # 惩罚不相关的键
                                            if score > 0:
                                                candidates.append((score, main_key))
                                        
                                        # 按分数排序，取前10个作为候选
                                        candidates.sort(reverse=True, key=lambda x: x[0])
                                        top_candidates = [c[1] for c in candidates[:10]]
                                        
                                        # 添加到替代键列表
                                        alternative_keys.update(top_candidates)
                                    
                                    print(f"  - 生成替代键数量: {len(alternative_keys)}")
                                    
                                    # 4. 尝试所有替代键
                                    for alt_key in alternative_keys:
                                        if alt_key in main_state_dict and alt_key != mapped_key:
                                            print(f"  - 尝试替代键: {alt_key}")
                                            try:
                                                main_weight = main_state_dict[alt_key]
                                                # 尝试形状匹配，包括转置
                                                if main_weight.shape == delta.shape:
                                                    print(f"  - 形状匹配，应用更新")
                                                    delta = delta.to(main_weight.dtype)
                                                    merged_dict[alt_key] = main_weight + delta
                                                    applied_layers += 1
                                                    print(f"  - 成功应用到替代键: {alt_key}")
                                                    success = True
                                                    break
                                                else:
                                                    # 尝试转置
                                                    if main_weight.shape == delta.T.shape:
                                                        print(f"  - 通过转置匹配形状")
                                                        delta = delta.T.to(main_weight.dtype)
                                                        merged_dict[alt_key] = main_weight + delta
                                                        applied_layers += 1
                                                        print(f"  - 成功应用到替代键: {alt_key}")
                                                        success = True
                                                        break
                                            except Exception as inner_e:
                                                print(f"  - 应用到替代键失败: {str(inner_e)}")
                                except Exception as e_last:
                                    print(f"  - 最后补救措施失败: {str(e_last)}")
                            
                            print(f"  - 跳过此层")
                            skipped_layers += 1
                            continue
                    except KeyError as e:
                        # 特别处理KeyError，提供更详细的错误信息和修复建议
                        missing_key = str(e).strip("'\"")
                        print(f"处理 {base_key} 时出错: KeyError('{missing_key}')")
                        print(f"  - 错误详情: 找不到键 '{missing_key}'")
                        
                        # 增强的错误处理，特别关注常见的失败情况
                        if 'txt_norm.weight' in missing_key or 'txt_norm' in missing_key:
                                print(f"  - 关键错误: 找不到文本归一化层 {missing_key}")
                                print(f"  - 自动搜索可能的文本归一化层替代...")
                                
                                # 搜索可能的文本归一化层
                                norm_candidates = []
                                for main_key in main_state_dict:
                                    # 查找包含norm和text相关词汇的键
                                    if ('norm' in main_key.lower() and 
                                        ('text' in main_key.lower() or 'txt' in main_key.lower())):
                                        # 计算相似度分数
                                        score = 0
                                        if 'weight' in main_key:
                                            score += 2
                                        if 'text' in main_key.lower():
                                            score += 1
                                        if 'norm' in main_key.lower():
                                            score += 1
                                        norm_candidates.append((score, main_key))
                                
                                # 按分数排序
                                norm_candidates.sort(reverse=True)
                                
                                if norm_candidates:
                                    best_candidate = norm_candidates[0][1]
                                    print(f"  - 找到最可能的替代层: {best_candidate}")
                                    print(f"  - 自动修复建议: 系统将尝试使用 '{best_candidate}' 替代 '{missing_key}'")
                                    # 尝试自动应用到找到的替代层
                                    try:
                                        if best_candidate in main_state_dict:
                                            print(f"  - 尝试将更新应用到替代层: {best_candidate}")
                                            # 这里需要重新计算delta，但我们已经有了delta
                                            # 直接尝试应用到替代层
                                            main_weight = main_state_dict[best_candidate]
                                            # 检查形状兼容性
                                            if main_weight.shape == delta.shape:
                                                delta = delta.to(main_weight.dtype)
                                                merged_dict[best_candidate] = main_weight + delta
                                                applied_layers += 1
                                                print(f"  - 成功应用到替代层: {best_candidate}")
                                                success = True
                                            else:
                                                print(f"  - 形状不匹配，无法应用到替代层")
                                    except Exception as inner_e:
                                        print(f"  - 应用到替代层失败: {str(inner_e)}")
                        elif 'cross_attn' in missing_key:
                                print(f"  - 关键错误: 找不到cross_attn相关层 {missing_key}")
                                print(f"  - 自动搜索可能的cross_attn替代层...")
                                
                                # 生成更全面的可能替代命名
                                alternatives = []
                                
                                # 1. 标准命名变体
                                alternatives.append(missing_key.replace('cross_attn', 'attn1'))
                                alternatives.append(missing_key.replace('cross_attn', 'attn2'))
                                alternatives.append(missing_key.replace('cross_attn', 'attn'))
                                
                                # 2. 增加更多常见的注意力层命名变体
                                alternatives.append(missing_key.replace('cross_attn', 'attention1'))
                                alternatives.append(missing_key.replace('cross_attn', 'attention2'))
                                alternatives.append(missing_key.replace('cross_attn', 'attention'))
                                alternatives.append(missing_key.replace('cross_attn', 'attn_qkv'))
                                alternatives.append(missing_key.replace('cross_attn', 'qkv_attn'))
                                
                                # 3. 增加block命名变体
                                alternatives.append(missing_key.replace('transformer_blocks', 'transformer_layers'))
                                alternatives.append(missing_key.replace('cross_attn', 'attn1').replace('transformer_blocks', 'transformer_layers'))
                                alternatives.append(missing_key.replace('cross_attn', 'attn2').replace('transformer_blocks', 'transformer_layers'))
                                alternatives.append(missing_key.replace('cross_attn', 'attention').replace('transformer_blocks', 'transformer_layers'))
                                
                                # 4. 尝试不同的命名约定
                                if '.block.' in missing_key:
                                    alternatives.append(missing_key.replace('.block.', '.layer.'))
                                if '.layer.' in missing_key:
                                    alternatives.append(missing_key.replace('.layer.', '.block.'))
                                
                                # 5. 添加文本相关的命名变体
                                alternatives.append(missing_key.replace('cross_attn', 'text_attn'))
                                alternatives.append(missing_key.replace('cross_attn', 'txt_attn'))
                                alternatives.append(missing_key.replace('cross_attn', 'text_attention'))
                                
                                # 6. 特殊模型架构的命名变体
                                # 对于某些SDXL架构变体
                                alternatives.append(missing_key.replace('cross_attn', 'attn_cross'))
                                alternatives.append(missing_key.replace('cross_attn', 'attention_cross'))
                                
                                # 基于层号和组件类型的智能搜索
                                # 提取层号和关键组件类型
                                import re
                                layer_match = re.search(r'blocks?\.([0-9]+)', missing_key)
                                
                                # 提取组件类型（如k/q/v/o/proj等）
                                component_match = re.search(r'cross_attn\.([a-z]+)', missing_key)
                                component = component_match.group(1) if component_match else None
                                
                                if layer_match:
                                    layer_num = layer_match.group(1)
                                    print(f"  - 提取到层号: {layer_num}")
                                    if component:
                                        print(f"  - 提取到组件类型: {component}")
                                    
                                    # 搜索相同层号的其他注意力层
                                    for main_key in main_state_dict:
                                        # 检查是否包含相同的层号
                                        if (f'blocks.{layer_num}' in main_key or f'layers.{layer_num}' in main_key or 
                                            # 也检查直接的层号数字模式
                                            re.search(r'blocks?\\.{0}$|layers?\\.{0}$|\\.{0}\\.'.format(layer_num), main_key)):
                                            
                                            # 优先级1: 包含attn和相同组件类型
                                            if component and 'attn' in main_key and component in main_key and 'cross_attn' not in main_key:
                                                print(f"  - 找到同层号且同组件类型的注意力层: {main_key}")
                                                alternatives.append(main_key)
                                            # 优先级2: 包含attn但不包含cross_attn
                                            elif 'attn' in main_key and 'cross_attn' not in main_key:
                                                alternatives.append(main_key)
                                            # 优先级3: 包含attention但不包含cross_attn
                                            elif 'attention' in main_key and 'cross_attn' not in main_key:
                                                alternatives.append(main_key)
                                
                                # 去重
                                alternatives = list(set(alternatives))
                                
                                # 搜索替代层
                                found_alt = False
                                for alt in alternatives:
                                    if alt in main_state_dict:
                                        print(f"  - 找到cross_attn替代层: {alt}")
                                        # 更新映射键并继续处理
                                        mapped_key = alt
                                        # 重置跳过计数，继续处理
                                        skipped_layers -= 1  # 抵消之前的增加
                                        success = True
                                        found_alt = True
                                        break
                                
                                # 如果标准替代失败，尝试更智能的搜索
                                if not found_alt:
                                    print(f"  - 执行更深度的cross_attn替代层搜索...")
                                    
                                    # 1. 提取关键部分用于搜索
                                    key_parts = missing_key.split('.')
                                    
                                    # 2. 提取有意义的部分
                                    meaningful_parts = []
                                    numeric_parts = []
                                    component_parts = []
                                    
                                    for part in key_parts:
                                        # 提取层号
                                        if part.isdigit():
                                            numeric_parts.append(part)
                                            meaningful_parts.append(part)
                                        # 提取组件类型
                                        elif part in ['k', 'q', 'v', 'o', 'proj', 'weight', 'bias', 'to_q', 'to_k', 'to_v', 'to_out']:
                                            component_parts.append(part)
                                            meaningful_parts.append(part)
                                        # 提取其他可能有意义的长部分
                                        elif len(part) > 4 and part not in ['cross', 'attn', 'transformer', 'blocks', 'layers']:
                                            meaningful_parts.append(part)
                                    
                                    print(f"  - 提取的关键部分: {meaningful_parts}")
                                    print(f"  - 提取的层号: {numeric_parts}")
                                    print(f"  - 提取的组件类型: {component_parts}")
                                    
                                    if meaningful_parts:
                                        # 3. 搜索最匹配的层，使用改进的评分算法
                                        best_matches = []
                                        
                                        for main_key in main_state_dict:
                                            match_score = 0
                                            
                                            # 基于关键部分匹配
                                            for part in meaningful_parts:
                                                if part in main_key:
                                                    # 对不同类型的匹配赋予不同权重
                                                    if part.isdigit():
                                                        match_score += 3  # 层号匹配权重最高
                                                    elif part in component_parts:
                                                        match_score += 2  # 组件类型匹配权重次之
                                                    else:
                                                        match_score += 1  # 其他部分匹配权重
                                                
                                            # 架构特征匹配
                                            # 优先考虑包含attn但不包含cross_attn的键
                                            if 'attn' in main_key and 'cross_attn' not in main_key:
                                                match_score += 4
                                            # 其次考虑包含attention但不包含cross_attn的键
                                            elif 'attention' in main_key and 'cross_attn' not in main_key:
                                                match_score += 3
                                            # 再次考虑包含qkv相关的键
                                            elif any(comp in main_key for comp in ['qkv', 'to_q', 'to_k', 'to_v']):
                                                match_score += 2
                                            # 避免完全不相关的层
                                            elif 'mlp' in main_key or 'norm' in main_key or 'conv' in main_key:
                                                match_score -= 5
                                                
                                            # 只有分数足够高的才加入候选
                                            if match_score > 5:  # 提高阈值，确保质量
                                                best_matches.append((match_score, main_key))
                                        
                                        # 按匹配分数排序
                                        best_matches.sort(reverse=True)
                                        
                                        # 尝试前3个最佳匹配，而不是只尝试1个
                                        for score, best_alt in best_matches[:3]:
                                            print(f"  - 找到深度匹配替代层(分数: {score}): {best_alt}")
                                            
                                            # 检查该层是否存在且形状兼容
                                            if best_alt in main_state_dict:
                                                try:
                                                    main_weight = main_state_dict[best_alt]
                                                    # 先检查直接形状匹配
                                                    if main_weight.shape == delta.shape:
                                                        mapped_key = best_alt
                                                        skipped_layers -= 1
                                                        success = True
                                                        print(f"  - 形状匹配成功，使用替代层: {best_alt}")
                                                        break
                                                    # 再检查转置后是否匹配
                                                    elif hasattr(delta, 'T') and main_weight.shape == delta.T.shape:
                                                        mapped_key = best_alt
                                                        skipped_layers -= 1
                                                        success = True
                                                        print(f"  - 转置后形状匹配成功，使用替代层: {best_alt}")
                                                        break
                                                except Exception as inner_e:
                                                    print(f"  - 检查替代层时出错: {str(inner_e)}")
                                            
                                            # 如果找到可用的替代层，跳出循环
                                            if success:
                                                break
                            
                        print(f"  - 建议解决方案:")
                        print(f"    1. 尝试使用与LoRA模型训练时相同架构的基础模型")
                        print(f"    2. 检查模型架构是否包含文本归一化层，可能使用了不同的命名")
                        print(f"    3. 继续处理其他层，跳过不兼容的部分")
                        
                        print(f"  - 可能原因: LORA模型与主模型架构不兼容")
                        skipped_layers += 1
                        # 继续处理下一层，不要中断整个过程
                        continue
                    except Exception as e:
                        print(f"处理 {key} 时出错: {str(e)}")
                        print(f"  - 错误详情: {repr(e)}")
                        print(f"  - 错误类型: {type(e).__name__}")
                        
                        # 检查是否是形状不匹配问题
                        if 'shape' in str(e).lower() or 'dimension' in str(e).lower():
                            print(f"  - 这可能是权重形状不匹配问题")
                            print(f"  - 建议检查模型架构兼容性")
                        
                        skipped_layers += 1
                        # 继续处理下一层，不要中断整个过程
                        continue
                else:
                    # 优化警告信息，提供更多上下文和尝试过的策略
                    print(f"警告: 主模型中未找到对应的层: {key}")
                    print(f"  - 尝试了以下映射策略但未找到匹配: 基础映射(13种模式)、复杂块映射(transformer_blocks变体)、移除模式映射(add_/img_前缀)、相似层映射")
                    print(f"  - 可能原因: 主模型与LoRA模型架构不兼容，或LoRA针对不同的模型变体训练")
                    print(f"  - 建议: 检查主模型和LoRA模型是否兼容，或尝试使用相同架构的模型")
                    skipped_layers += 1
                
                # 标记为已处理
                processed_keys.add(key)
                processed_keys.add(up_key)
            
            # 定期释放内存
            if key_idx % 50 == 0:
                print("正在执行内存回收...")
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                memory_usage = self._get_memory_usage()
                print(f"当前内存使用: {memory_usage:.2f} MB")
        
        # 记录最终统计信息
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print(f"LORA融合操作完成")
        print(f"- 成功应用: {applied_layers} 个层")
        print(f"- 跳过: {skipped_layers} 个层")
        print(f"- 总耗时: {elapsed_time:.2f} 秒")
        # 修复除零错误
        if applied_layers + skipped_layers > 0:
            print(f"- 平均每层处理时间: {(elapsed_time / (applied_layers + skipped_layers)):.3f} 秒")
        else:
            print("- 未处理任何层")
        print("=" * 80)
        
        # 保存融合后的模型
        final_output_path = output_path
        try:
            # 调用保存方法，可能返回备用路径
            final_output_path = self._save_model_safely(merged_dict, output_path)
        except Exception as e:
            error_msg = f"保存融合模型失败: {str(e)}"
            print(error_msg)
            print("建议解决方案:")
            print("1. 检查输出目录是否有写入权限")
            print("2. 确保磁盘有足够空间")
            print("3. 尝试使用不同的输出目录")
            print("4. 关闭可能正在占用该文件的其他程序")
            raise RuntimeError(error_msg)
        finally:
            # 确保释放内存，无论保存是否成功
            print("执行最终内存回收...")
            if 'main_state_dict' in locals():
                main_state_dict.clear()
            if 'lora_state_dict' in locals():
                lora_state_dict.clear()
            if 'merged_dict' in locals():
                merged_dict.clear()
            # 强制垃圾回收
            for _ in range(3):  # 多轮回收
                gc.collect()
                time.sleep(0.1)
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        
        return final_output_path
    
    def process_model(self, operation_type: str, main_model: str, output_file: str, 
                     model_files: Optional[str] = None, merge_mode: str = "update", 
                     quantize_bits: str = "none", lora_file: Optional[str] = None, 
                     lora_scale: float = 1.0, output_directory: Optional[str] = None,
                     use_sharded_processing: bool = True, shard_size_mb: int = 512,
                     auto_adjust_shard: bool = True,
                     enable_detailed_logging: bool = False,
                     overwrite_existing: bool = True) -> Tuple[str]:
        """处理模型的主函数，根据操作类型执行不同的功能"""
        try:
            # 打印详细的参数信息
            print("="*80)
            print(f"[处理开始] 操作类型: {operation_type}")
            print(f"[环境信息] 当前工作目录: {os.getcwd()}")
            print(f"[参数信息] main_model: '{main_model}'")
            print(f"[参数信息] output_file: '{output_file}'")
            print(f"[参数信息] output_directory: '{output_directory}'")
            print(f"[参数信息] lora_file: '{lora_file}'")
            print(f"[参数信息] lora_scale: {lora_scale}")
            print(f"[参数信息] quantize_bits: {quantize_bits}")
            print(f"[参数信息] merge_mode: {merge_mode}")
            print(f"[参数信息] use_sharded_processing: {use_sharded_processing}")
            print(f"[参数信息] shard_size_mb: {shard_size_mb}")
            print(f"[参数信息] auto_adjust_shard: {auto_adjust_shard}")
            print(f"[参数信息] enable_detailed_logging: {enable_detailed_logging}")
            print("="*80)
            
            # 准备输出路径
            output_path = self._prepare_output_path(output_directory, output_file)
            
            # 检查是否覆盖现有文件
            if os.path.exists(output_path) and not overwrite_existing:
                raise FileExistsError(f"目标文件已存在且不允许覆盖: {output_path}")
            
            # 根据操作类型执行相应的功能
            if operation_type == "merge":
                # 合并模型功能
                print("[操作处理] 开始执行模型合并操作...")
                
                # 收集模型文件路径
                file_paths = []
                
                # 从多行输入获取
                if model_files and model_files.strip():
                    file_paths = [path.strip() for path in model_files.strip().split("\n") if path.strip()]
                
                # 如果没有从多行输入获取到路径，使用主模型路径
                if not file_paths and main_model:
                    file_paths.append(main_model)
                    print("注意: 从主模型路径获取了一个模型文件进行合并")
                
                if not file_paths:
                    raise ValueError("请提供要合并的模型文件路径")
                
                # 优化：如果只有单个文件且需要量化，提示用户直接使用quantize操作类型
                if len(file_paths) == 1 and quantize_bits != "none":
                    print("[提示] 检测到只有单个模型文件且需要量化")
                    print("[提示] 建议: 您可以直接选择'quantize'操作类型以获得更好的性能")
                    print("[提示] 当前将继续执行合并+量化流程")
                
                # 验证所有文件是否存在，并尝试自动检测分片文件
                valid_files = []
                processed_paths = set()
                
                for file_path in file_paths:
                    try:
                        valid_path = self._find_file(file_path)
                        # 避免重复添加
                        if valid_path not in processed_paths:
                            valid_files.append(valid_path)
                            processed_paths.add(valid_path)
                            
                            # 尝试自动检测分片文件（如diffusion_pytorch_model-00001-of-00005.safetensors格式）
                            dir_path = os.path.dirname(valid_path)
                            file_name = os.path.basename(valid_path)
                            
                            # 检查文件名是否匹配分片模式
                            import re
                            shard_pattern = r'(.*)-(\d+)-of-(\d+)\.safetensors'
                            match = re.match(shard_pattern, file_name)
                            
                            if match:
                                base_name = match.group(1)
                                total_shards = int(match.group(3))
                                
                                print(f"检测到分片模型: {base_name}, 共 {total_shards} 个分片")
                                
                                # 尝试加载所有其他分片
                                for i in range(1, total_shards + 1):
                                    shard_index_str = str(i).zfill(5)  # 格式化为5位数字
                                    shard_file_name = f"{base_name}-{shard_index_str}-of-{str(total_shards).zfill(5)}.safetensors"
                                    shard_path = os.path.join(dir_path, shard_file_name)
                                    
                                    # 如果该分片尚未处理
                                    if shard_path not in processed_paths:
                                        try:
                                            # 直接检查文件是否存在，避免重复搜索
                                            if os.path.exists(shard_path):
                                                print(f"自动找到分片文件: {shard_path}")
                                                valid_files.append(shard_path)
                                                processed_paths.add(shard_path)
                                            # 也尝试_normalize_path以防路径格式问题
                                            elif os.path.exists(self._normalize_path(shard_path)):
                                                norm_path = self._normalize_path(shard_path)
                                                print(f"自动找到分片文件(规范化路径): {norm_path}")
                                                valid_files.append(norm_path)
                                                processed_paths.add(norm_path)
                                        except Exception as e:
                                            print(f"警告: 尝试加载分片文件 {shard_file_name} 时出错: {str(e)}")
                    except FileNotFoundError as e:
                        print(f"警告: {str(e)}")
                
                if not valid_files:
                    raise ValueError("未找到有效的模型文件")
                
                print(f"找到 {len(valid_files)} 个有效模型文件进行合并")
                
                # 执行合并
                if use_sharded_processing and len(valid_files) > 1:
                    print(f"[分片模式] 启用分片处理，文件数量: {len(valid_files)}")
                    merged_path = self._sharded_merge(valid_files, output_path, merge_mode, 
                                                     shard_size_mb, auto_adjust_shard, 
                                                     enable_detailed_logging)
                else:
                    # 简单合并（单个文件或禁用分片处理）
                    print("使用简单合并模式")
                    merged_dict = {}
                    total_files = len(valid_files)
                    
                    for file_idx, file_path in enumerate(valid_files, 1):
                        print(f"加载文件 {file_idx}/{total_files}: {file_path}")
                        state_dict = load_file(file_path)
                        
                        if merge_mode == "update":
                            merged_dict.update(state_dict)
                        else:
                            merged_dict = state_dict
                            
                        # 记录进度并释放内存
                        progress = (file_idx / total_files) * 100
                        print(f"合并进度: {progress:.1f}% - 已处理 {file_idx}/{total_files} 个文件")
                        state_dict.clear()
                        gc.collect()
                    
                    # 调用保存方法，可能返回备用路径
                    save_result = self._save_model_safely(merged_dict, output_path)
                    if save_result != output_path:
                        print(f"注意: 模型已保存到备用路径: {save_result}")
                        output_path = save_result
                    merged_path = output_path
                    merged_dict.clear()
                
                # 如果需要在合并后进行量化
                if quantize_bits != "none":
                    print(f"[操作处理] 合并完成后执行量化操作，目标方式: {quantize_bits}")
                    # 使用临时路径进行量化，避免文件锁定问题
                    temp_quant_path = output_path.replace('.safetensors', f'_quant_temp_{int(time.time())}.safetensors')
                    print(f"[操作处理] 使用临时路径进行量化: {temp_quant_path}")
                    quant_result = self._quantize_model(merged_path, temp_quant_path, quantize_bits)
                    
                    # 如果量化成功，尝试将临时文件移动到最终路径
                    if os.path.exists(temp_quant_path):
                        try:
                            print(f"[操作处理] 将量化模型从临时路径移动到最终路径")
                            # 先删除目标文件（如果存在）
                            if os.path.exists(output_path):
                                try:
                                    os.remove(output_path)
                                except:
                                    print(f"[警告] 无法删除原文件: {output_path}")
                            
                            # 使用shutil.move进行跨设备安全移动
                            import shutil
                            shutil.move(temp_quant_path, output_path)
                            print(f"[操作处理] 量化模型成功保存到: {output_path}")
                            return output_path,
                        except Exception as e:
                            print(f"[警告] 无法将量化模型移动到最终路径: {str(e)}")
                            print(f"[提示] 量化模型已保存在临时路径: {temp_quant_path}")
                            return temp_quant_path,
                    
                    return quant_result,
                
                return merged_path,
                
            elif operation_type == "quantize":
                # 量化模型功能
                print("[操作处理] 开始执行模型量化操作...")
                # 查找主模型文件
                main_model_path = self._find_file(main_model)
                return self._quantize_model(main_model_path, output_path, quantize_bits),
                
            elif operation_type == "merge_lora":
                # 融合LORA模型功能
                print("[操作处理] 开始执行LORA融合操作...")
                # 查找主模型和LORA文件
                main_model_path = self._find_file(main_model)
                lora_path = self._find_file(lora_file)
                return self._merge_lora(main_model_path, lora_path, output_path, lora_scale, enable_detailed_logging)
                
            else:
                raise ValueError(f"不支持的操作类型: {operation_type}")
                
        except Exception as e:
            import traceback
            print(f"[错误] 处理过程出错: {str(e)}")
            print("[错误] 详细错误堆栈:")
            traceback.print_exc()
            # 返回错误信息而不是抛出异常，避免ComfyUI崩溃
            return (f"错误: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "ModelMergerNode": ModelMergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergerNode": "增强模型处理器"
}