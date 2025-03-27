import os

import torch
import torch.nn as nn

from dataloader import create_dataloaders
from model import PatchEmbedding


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
patch_size = 16
embedding_dim = 768

current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'food/train')
test_dir = os.path.join(current_dir, 'food/test')

train_data, test_data, label = create_dataloaders(train_dir, test_dir, batch_size=batch_size)

# Images to patches
patchify = PatchEmbedding(patch_size=patch_size, embedding_dim=embedding_dim)
imgs, label = next(iter(train_data))
_, _, H, W = imgs.shape
patches = patchify(imgs)    #(batch_size, patch_nums, embedding_dim)

# Class token
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dim), requires_grad=True)    #(batch_size, 1, embedding_dim)

# Class token + Image patches
patch_embedding = torch.cat((class_token, patches), dim=1)   #(batch_size, patch_num+1, embedding_dim)

# Position embedding
patch_num = int((H*W) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, patch_num+1, embedding_dim), requires_grad=True)

total_embedding = patch_embedding + position_embedding
