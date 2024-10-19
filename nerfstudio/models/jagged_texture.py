import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from gstex_cuda.texture_sample import texture_sample
from gstex_cuda._torch_impl import quat_to_rotmat, normalized_quat_to_rotmat, sample_texture

def texture_dims_to_int_coords(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs, hws, dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = local_idxs // query_dims[:,1]
    vv = local_idxs % query_dims[:,1]
    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv

def texture_dims_to_query(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs, hws, dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = (local_idxs // query_dims[:,1]).float() / query_dims[:,0].float()
    vv = (local_idxs % query_dims[:,1]).float() / query_dims[:,1].float()
    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv

class JaggedTexture(nn.Module):
    def __init__(self, texture_dims, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        # self.temp = nn.Parameter(torch.zeros(1,))
        self.texture = nn.Parameter(torch.zeros(1,self.out_dim))
        self.register_buffer("texture_dims", texture_dims)
        self.init_from_dims(texture_dims, initial=True)
    
    def get_texture(self):
        return self.texture[:self.total_size,:]
    
    def reset(self):
        with torch.no_grad():
            self.texture.data = torch.zeros_like(self.texture.data)
            # self.texture.data = 2 * torch.rand_like(self.texture.data) - 1

    def adjust_texture_size(self, new_size):
        diff = new_size - self.texture.shape[0]
        if diff <= 0:
            return
        self.texture = nn.Parameter(
            torch.cat(
                [
                    self.texture.detach(),
                    torch.zeros((diff, self.texture.shape[1]), dtype=self.texture.dtype, device=self.texture.device)
                ], dim=0
            )
        )

    def cull(self, cull_mask):
        idxs = torch.arange(self.texture_dims.shape[0], dtype=torch.int64, device=self.texture_dims.device)
        old_hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        ids = torch.repeat_interleave(idxs, old_hws, dim=0)
        texture_cull_mask = cull_mask[ids]
        new_size = self.total_size - texture_cull_mask.sum().item()
        self.adjust_texture_size(new_size)
        with torch.no_grad():
            self.texture.data[:new_size,:] = self.texture.data[:self.total_size,:][~texture_cull_mask,:]
        
        self.texture_dims = self.texture_dims[~cull_mask]
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        self.total_size = torch.sum(self.texture_dims[:,0] * self.texture_dims[:,1]).item()
        assert self.total_size == new_size

    def dup_and_split(self, dup_mask, split_mask, samps):
        idxs = torch.arange(self.texture_dims.shape[0], dtype=torch.int64, device=self.texture_dims.device)
        old_hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        ids = torch.repeat_interleave(idxs, old_hws, dim=0)
        texture_dup_mask = dup_mask[ids]
        texture_split_mask = split_mask[ids]

        new_size = self.total_size + texture_dup_mask.sum().item() + samps * texture_split_mask.sum().item()
        self.adjust_texture_size(new_size)

        with torch.no_grad():
            texture = self.get_texture().data
            self.texture.data[:new_size,:] = torch.cat(
                [
                    texture,
                    texture[texture_dup_mask,:],
                    texture[texture_split_mask,:].repeat(samps, 1),
                ],
                dim=0
            )
    
        self.texture_dims = torch.cat(
            [
                self.texture_dims,
                self.texture_dims[dup_mask],
                self.texture_dims[split_mask].repeat(samps, 1),
            ],
            dim=0
        )
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        self.total_size = torch.sum(self.texture_dims[:,0] * self.texture_dims[:,1]).item()
        assert self.total_size == new_size

    def init_from_dims(self, texture_dims, initial=False):
        if not initial:
            assert texture_dims.shape[0] == self.texture_dims.shape[0]
        
        total_size = torch.sum(texture_dims[:,0] * texture_dims[:,1]).item()
        with torch.no_grad():
            if initial:
                self.max_size = total_size
                self.texture = nn.Parameter(torch.zeros(total_size, self.out_dim, device=self.texture.device))
            else:
                idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
                hws = texture_dims[:,0] * texture_dims[:,1]
                ids = torch.repeat_interleave(idxs, hws, dim=0)
                self.texture_dims = self.texture_dims.to(texture_dims.device)
                query_dims = self.texture_dims[ids,:]

                _, uv = texture_dims_to_query(texture_dims)
                # new_texture = sample_texture(query_dims, self.texture, uv)
                texture_info = (1, 1, self.texture.shape[-1])
                use_torch_impl = False
                if "cuda" not in str(self.texture.device):
                    use_torch_impl = True
                new_texture = texture_sample(texture_info, query_dims, self.get_texture(), uv, use_torch_impl=use_torch_impl)
                self.adjust_texture_size(total_size)
                self.texture.data[:total_size,:] = new_texture.detach()
        
        self.texture_dims = texture_dims
        self.total_size = total_size

            
    
    def forward(self, x, query_dims, detached=False):
        y = torch.clamp(x, min=0.0, max=1.0)

        if detached:
            texture = self.get_texture().detach()
        else:
            texture = self.get_texture()

        y = sample_texture(query_dims, texture, y)
        return y