import torch
import torchvision
import torch.nn as nn
# model = torchvision.models.vit_b_16()
#
# in_c = model.heads[0].in_features
# model.heads[0] = nn.Linear(in_c, 9)


model = torchvision.models.swin_t()
in_c = model.head.in_features
model.heads = nn.Linear(in_c, 9)
print(model)