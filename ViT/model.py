import cv2
import numpy as np

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module) :
    
    def __init__ (self, patch_size, embedding_dim, in_channels=3) :
        
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, img) :

        patch = self.patcher(img)
        flat = self.flatten(patch)
        return flat.permute(0, 2, 1)
    
