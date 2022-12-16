# --------------------------------------------------------
# The model has been implmented as a modification of the MAE model code found here: 
# https://github.com/facebookresearch/mae
# In some methods, only minor moodifications have been doen to the original MAE code.
# Some methods have not been chnaged at all and simply copied from the original code.
# A copy of the licenese for the MAE code can be found in the thirdparty folder. 
# 
# Uses timm==0.5.4 for model components. 

# References: 
# https://github.com/facebookresearch/mae
# https://github.com/rwightman/pytorch-image-models
# -------------------------------------------------------- 

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from thirdparty.pos_embed import get_2d_sincos_pos_embed


class ObjectViTAE(nn.Module):
    """Transformer-based autoencoder for object-centric learning. 
    Built as a modification of MAE.

    No changes at all have been done to the following methods from MAE:
    - _init_weights
    - patchify
    - unpatchify
    - random_masking

    Other methods do contain some changes, altough sometimes only minor.
    """
    def __init__(self, args):
        super().__init__()
        # Define some parameters for amount of slots and 
        self.num_slots = args.num_slots
        self.res = args.img_resolution
        self.num_channels = args.img_channels
        self.patch_size = args.patch_size
        self.scale = args.embed_dim ** -0.5

        # Embedding and class tokens for object segmentation
        self.patch_embed = PatchEmbed(args.img_resolution, args.patch_size, args.img_channels, args.embed_dim)
        args.num_patches = self.patch_embed.num_patches
        self.cls_tokens = nn.Parameter(torch.zeros(1, args.num_slots, args.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, args.num_patches, args.embed_dim), requires_grad=False)  # sin-cos embedding

        # ViT encoder
        self.blocks = nn.ModuleList([
            Block(args.embed_dim, args.num_heads, args.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(args.depth)])
        self.norm = nn.LayerNorm(args.embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(args.embed_dim + 1, args.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, args.num_patches, args.decoder_embed_dim), requires_grad=False)  # sin-cos embedding

        # ViT decoder
        self.decoder_blocks = nn.ModuleList([
            Block(args.decoder_embed_dim, args.decoder_num_heads, args.mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(args.decoder_depth)])

        # Predicting reconstructions and output masks (alpha channels)
        self.decoder_norm = nn.LayerNorm(args.decoder_embed_dim)
        self.decoder_pred = nn.Linear(args.decoder_embed_dim, args.patch_size**2 * (args.img_channels+1), bias=True) # decoder to patch

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # sin-cos embedding (not for class tokens)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # As in MAE
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initializing cls tokens with small standard deviation
        torch.nn.init.normal_(self.cls_tokens, std=.002) 
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def unpatchify_alpha(self, x):
        """
        Copied from original implmentation of unpatchify in MAE, 
        but changed channels to 4 to account for the alpha channel.

        x: (N, L, patch_size**2 * 4)
        imgs: (N, 4, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(-1, self.num_slots, h, w, p, p, 4))
        x = torch.einsum('nshwpqc->nschpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.num_slots, 4, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, noise=0.0):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # Masking as in MAE.
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Create the k cls tokens 
        cls_tokens = self.cls_tokens
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)

        # Optional addaition of noise to cls tokens before appending
        if noise > 0.0:
            cls_tokens = cls_tokens + noise * torch.randn_like(cls_tokens)

        # Append the cls_tokens 
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore, mask

    def forward_object_function(self, x):
        # Separate cls tokens from the sequence 
        cls_tokens = x[:, :self.num_slots, :]
        x = x[:, self.num_slots:, :]

        # Scaled dot product between embeddings of patches and cls tokens
        attn_logits = torch.bmm(x, cls_tokens.permute(0, 2, 1)) * self.scale

        # Calculate log-softmax as we need logs of softmax later
        attn_logits = attn_logits.log_softmax(dim=-1) 
        # Get the softmax for weighted mean
        attn = attn_logits.exp() 
        
        # Compute weights for weighted mean
        w_attn = attn / (attn.sum(dim=1, keepdims=True) + 1e-8)
        
        # Slots as weighted mean
        slots = torch.bmm(w_attn.permute(0,2,1), x)

        return slots, attn_logits, attn


    def forward_broadcast_module(self, slots, attn_logits, ids_restore):
        # Repeat ids for restoring to be one per slot
        # as we decode slots separately
        ids_restore = ids_restore.unsqueeze(1).repeat(1,self.num_slots,1)
        ids_restore = ids_restore.reshape(-1,ids_restore.size(2))

        # Reshape attention masks for concatenating to broadcasted slots
        attn_logits = attn_logits.permute(0,2,1)
        attn_logits = attn_logits.reshape(-1,attn_logits.size(-1),1)
       
        # Broadcast the slots
        slots = slots.reshape(-1,1,slots.size(-1))
        slots = slots.repeat(1, attn_logits.size(-2),1)
        
        # Concatenate logs of patch masks to the broadcasted slots
        x = torch.cat((slots,attn_logits), dim=-1)

        # Embed 
        x = self.decoder_embed(x)

        # Add mask tokens as in MAE
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1) 
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  

        return x
        

    def forward_decoder(self, x):
        # Add position embeddings
        x = x + self.decoder_pos_embed 

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict reconstruction and alpha channel
        x = self.decoder_pred(x)

        # Unpatchify, resulting in reconstructions and output masks (alpha channel)
        x = self.unpatchify_alpha(x)

        # Split into alpha channels (masks) and RGB channels
        out_mask = x[:, :, self.num_channels:, :, :]
        pred_ind = x[:, :, :self.num_channels, :, :]

        # Softmax normalize alpha channels
        out_mask = out_mask.softmax(dim=1)

        # Calucalate full reconstruction as weighted sum 
        pred = (out_mask * pred_ind).sum(dim=1)
    
        return pred, pred_ind, out_mask


    def forward(self, imgs, mask_ratio=0.75, noise=0.0):
        # ViT encoder
        x, ids_restore, mask = self.forward_encoder(imgs, mask_ratio, noise=noise)
        
        # Object function
        slots, attn_logits, attn = self.forward_object_function(x)

        # Broadcast module
        x = self.forward_broadcast_module(slots, attn_logits, ids_restore)

        # ViT decoder
        pred, pred_ind, out_mask = self.forward_decoder(x)  

        return pred, out_mask, pred_ind, slots, attn, mask, ids_restore