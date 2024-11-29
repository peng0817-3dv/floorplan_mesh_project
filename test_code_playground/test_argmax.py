import numpy as np

a = np.arange(18).reshape(2,3,3) + 10
print(a)
b = a.argmax(-1)
print(b)
c = b.reshape(-1)
print(c)

import torch
import torch.nn.functional as F

input = torch.randn(3, 5, requires_grad=True,dtype=torch.float32)

target = torch.randint(5, (3,), dtype=torch.int64)

output = F.cross_entropy(input, target)
_, pred = torch.max(input, dim=-1)
sum = (pred == target).sum()
print(sum)