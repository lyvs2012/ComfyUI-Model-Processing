"""
ComfyUI 模型合并插件
用于合并多个 safetensors 模型文件
"""

# 导入我们的节点模块
from .model_merger_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 确保导出所有必要的映射
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']