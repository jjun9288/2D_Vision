import cv2
import numpy as np

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module) :
    def __init__ (self, patch_size, embedding_dim, in_channels=3) : 
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x) :
        patch = self.patcher(x)
        flat = self.flatten(patch)
        return flat.permute(0, 2, 1)

h
class MSABlock(nn.Module) :
    def __init__(self, embedding_dim, head_num) :
        super().__init__()
        # Norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # Multi-Head Attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head_num, batch_first=True)

    def forward(self, x) :
        x = self.layer_norm(x)
        attn, _ = self.multihead_attn(query=x, key=x, value=x)
        return attn
    

class MLPBlock(nn.Module) :
    def __init__(self, embedding_dim, MLP_size, dropout_rate=0.1) :
        super().__init__()
        # Norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=MLP_size),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=MLP_size, out_features=embedding_dim),
            nn.Dropout(p=dropout_rate)
        )
    
    def forward(self, x) :
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    

class TransformerEncoderBlock(nn.Module) : 
    def __init__(self, embedding_dim, num_heads, MLP_size, dropout_rate=0.1) :
        super().__init__()
        # MSA block
        self.msa_block = MSABlock(embedding_dim=embedding_dim, num_heads=num_heads)
        # MLP block
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, MLP_size=MLP_size)

    def forward(self ,x) :
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module) :
    def __init__(self, class_num, img_size=224, patch_size=16, embedding_dim=768, MLP_size=3072, head_num=12) :
        super().__init__()
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embedding_dim=embedding_dim)
        # Total number of patches
        self.patch_num = (img_size * img_size) / patch_size**2
        # Class token (learnable)
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        # Position embedding (learnable)
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num+1, embedding_dim), requires_grad = True)
        # Dropout layer
        self.embedding_dropout = nn.Dropout(p=0.1)
        # Transformer encoder block
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=head_num, MLP_size=MLP_size)]
            )
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=class_num)
            )
    
    def forward(self, x) :
        batch_size = x.shape[0]
        # Create class token
        class_token = self.class_token.expand(batch_size, -1, -1)
        # Create patch embedding
        x = self.patch_embedding(x)
        # class token + patch embedding
        x = torch.cat((class_token, x), dim=1)
        # Add position embedding
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x