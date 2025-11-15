import sys
import traceback

print("开始测试导入...")

try:
    # 尝试导入我们的模块
    from model_merger_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("✓ 成功导入NODE_CLASS_MAPPINGS和NODE_DISPLAY_NAME_MAPPINGS")
    print(f"可用节点: {list(NODE_CLASS_MAPPINGS.keys())}")
    print(f"显示名称: {list(NODE_DISPLAY_NAME_MAPPINGS.values())}")
except Exception as e:
    print(f"✗ 导入失败: {str(e)}")
    print("详细错误堆栈:")
    traceback.print_exc()

print("\n测试完成。")
