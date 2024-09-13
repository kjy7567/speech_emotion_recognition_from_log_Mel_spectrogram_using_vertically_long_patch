# Took from vit-pytorch github repository
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# Some modification exist

import math
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# CreateCoords took from CoordConv github repository
# https://github.com/walsvid/CoordConv
# Some modifications exist

def CreateCoords(max_bs=32, x_dim=64, y_dim=64, with_r=False, skiptile=False):
    """Add coords to a tensor"""
    # self.x_dim = x_dim
    # self.y_dim = y_dim
    # self.with_r = with_r
    # self.skiptile = skiptile

    batch_size_tensor = max_bs  # Get batch size
                                # If you want larger batch, change max_bs

    xx_ones = torch.ones([1, x_dim], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = torch.arange(y_dim, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = torch.ones([1, y_dim], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = torch.arange(x_dim, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 2, 1)
    yy_channel = yy_channel.permute(0, 3, 2, 1)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    coords = torch.cat([xx_channel, yy_channel], dim=1)
    coords = coords.repeat(batch_size_tensor, 1, 1, 1)

    return coords.to('cuda')

def sinusoidal_pe(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.to('cuda')

class CustomDataset(Dataset):
    def __init__(self, img_list, trg_list):
        self.img = img_list
        self.trg = trg_list
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img = plt.imread(self.img[idx])[:,:,:1]
        img = self.transforms(img)
        trg = self.trg[idx]
        return {"img": img, "trg": trg}

class Teacher(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., max_bs = 32):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.InstanceNorm2d(1),
            nn.GELU(),
        )

        self.coords = CreateCoords(max_bs=max_bs, x_dim=image_width, y_dim=image_height)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = (channels+2) * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Linear(dim, num_classes)

    def encoder(self, img):
        x = self.conv_stem(img)
        # x = img
        x = torch.cat((x,self.coords[:x.size(0)]), dim=1)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        return x
    
    def decoder(self, x):
        y = x
        x = self.mlp_head(x)

        return x, y

    def forward(self, img):
        x = self.encoder(img)
        x, y = self.decoder(x)
        return x, y

class Student(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., max_bs = 32):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = (channels+0) * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Linear(dim, num_classes)

    def encoder(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        return x
    
    def decoder(self, x):
        y = x
        x = self.mlp_head(x)

        return x, y

    def forward(self, img):
        x = self.encoder(img)
        x, y = self.decoder(x)
        return x, y