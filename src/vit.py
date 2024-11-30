# Implementation of Vision Transformer and Performer-F Variant(s).
import os
import time
from functools import partial
import traceback
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

ALLOWED_MODELS = ('softmax', 'relu', 'exp', 'learn')
LOG_COLS = (
    'Model', 'Epoch', 'Training Time', 'Inference Time', 'Train Loss', 
    'Validation Loss', 'Train Accuracy', 'Validation Accuracy'
)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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

class LearnableAttention(nn.Module):
    def __init__(self, dim, la_depth=1, la_exp=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(la_depth):
            self.layers.append(nn.Linear(dim, dim))
            self.layers.append(nn.LayerNorm(dim))
            self.layers.append(nn.GELU())    
        self.layers.append(nn.Linear(dim, dim))
        self.activ = torch.exp if la_exp else F.relu
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.activ(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='softmax', **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Identity()
        
        assert attn_type in ALLOWED_MODELS, f'attn_type must be one of {ALLOWED_MODELS}.'

        if attn_type == 'softmax':
            self.attention = partial(self.compute_attention, func=torch.exp, before=False, scale=dim_head**(-1/2), eps=1e-8)
        elif attn_type == 'relu':
            self.attention = partial(self.compute_attention, func=F.relu,    before=True,  scale=None,             eps=1e-8)
        elif attn_type == 'exp':
            self.attention = partial(self.compute_attention, func=torch.exp, before=True,  scale=None,             eps=1e-8)
        elif attn_type == 'learn':
            self.my_learn = LearnableAttention(dim=dim_head, **kwargs)
            self.attention = partial(self.compute_attention, func=self.my_learn, before=True, scale=None, eps=1e-8)

    def compute_attention(self, q, k, func, before=True, scale=None, eps=None):
        if before:
            # function -> outer product (ReLU, exp)
            numer = torch.matmul(func(q), func(k).transpose(-1, -2))
        else:
            # outer product -> optional scaling -> function (softmax)
            numer = torch.matmul(q, k.transpose(-1, -2))
            if scale is not None:
                numer *= scale
            numer = func(numer)
        denom = numer.sum(dim=-1, keepdim=True)
        # avoiding division by zero
        if eps is not None: denom += eps
        # normalizing attention matrix
        return numer / denom

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = self.attention(q, k)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, **kwargs),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
        pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., **kwargs):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, **kwargs)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class Trainer:
    def __init__(self, lr, gamma, **kwargs):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available()  else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        self.model = ViT(**kwargs).to(self.device)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma) if gamma is not None else None
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=gamma) if gamma is not None else None
        
        self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 0
        
        self.best_vloss = float('inf')

    def logger(self, *args, log_dir):
        try:
            if not args:
                f = open(log_dir, mode='w', newline='')
                csv.writer(f).writerow(LOG_COLS)
            else:
                f = open(log_dir, mode='a', newline='')
                csv.writer(f).writerow(args)

        except:
            traceback.print_exc()
        
        finally:
            f.close()

    def save_checkpoint(self, checkpoint_dir, epoch, best_vloss):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(), 
                'optimizer_state_dict': self.optimizer.state_dict(), 
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None, 
                'epoch': epoch+1, 'best_vloss': best_vloss,
            }, 
            checkpoint_dir
        )

    def load_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            return False
        
        checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=True)

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_vloss = checkpoint['best_vloss']
            return True
        except:
            return False
                
    def train_step(self, train_loader):
        self.model.train()
        running_loss = running_acc = 0.
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += (outputs.argmax(dim=1) == labels).float().mean().item()        
        avg_loss, avg_acc = running_loss/(i+1), running_acc/(i+1)
        return avg_loss, avg_acc

    def valid_step(self, valid_loader):
        self.model.eval()
        running_vloss = running_vacc = 0.
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(valid_loader):
                vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                voutputs = self.model(vinputs)
                vloss = self.criterion(voutputs, vlabels)

                running_vloss += vloss.item()
                running_vacc += (voutputs.argmax(dim=1) == vlabels).float().mean().item()
        avg_vloss, avg_vacc = running_vloss/(i+1), running_vacc/(i+1)
        return avg_vloss, avg_vacc

    def train_valid(self, epochs, train_loader, valid_loader, log_dir, checkpoint_dir, model_name, notebook):
        if notebook: from tqdm.notebook import tqdm
        else: from tqdm import tqdm

        hist = list()
        success = self.load_checkpoint(checkpoint_dir=checkpoint_dir)

        if not success: self.logger(log_dir=log_dir)

        for epoch in tqdm(range(self.start_epoch, self.start_epoch+epochs), desc='Epoch', leave=False, position=0):
            start_train_time = time.time()
            avg_loss, avg_acc = self.train_step(train_loader=train_loader)
            end_train_time = time.time()

            start_infer_time = time.time()
            avg_vloss, avg_vacc = self.valid_step(valid_loader=valid_loader)
            end_infer_time = time.time()

            if self.scheduler is not None: self.scheduler.step()

            train_time = end_train_time - start_train_time
            infer_time = end_infer_time - start_infer_time

            log_info = (model_name, epoch, train_time, infer_time, avg_loss, avg_vloss, avg_acc, avg_vacc)

            hist.append(log_info)
            self.logger(*log_info, log_dir=log_dir)

            if avg_vloss < self.best_vloss:
                self.save_checkpoint(checkpoint_dir=checkpoint_dir, epoch=epoch, best_vloss=avg_vloss)
                
                self.best_vloss = avg_vloss

        return hist

if __name__ == '__main__':
    pass
