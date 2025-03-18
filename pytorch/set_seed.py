import torch

cuda = torch.device("cuda")
x = torch.rand((3, 3), device=cuda)
y = torch.rand((3, 3), device=cuda)

print(x)
print(y)

seed = 32
torch.cuda.manual_seed(seed)
x = torch.rand((3, 3), device=cuda)
torch.cuda.manual_seed(seed)
y = torch.rand((3, 3), device=cuda)

print(x)
print(y)