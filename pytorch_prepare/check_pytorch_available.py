#! python3

import torch

# 基础验证
print(f"PyTorch版本: {torch.__version__}")
print(f"GPU可用状态: {torch.cuda.is_available()}")

# 张量计算验证
x = torch.rand(3, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = x.to(device)
    print(f"GPU计算示例:\n{y}")
else:
    print("CPU计算示例:\n", x * 2)