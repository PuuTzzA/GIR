import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from . import renderutils as ru
from .utils import *

class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)
class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.smooth = Smooth()
    
    def forward(self, x):
        return self.smooth(self.up_sample(x))

class EnvLight(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, max_res=512, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable

        # init an empty cubemap
        self.init_base = torch.nn.Parameter(
            torch.randn(1, 128, 64, 128, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )
        self.base_train = torch.nn.Parameter(
           torch.zeros(6, 512, 512, 3, dtype=torch.float32, device=self.device),
           requires_grad=self.trainable,
        )
        self.base = torch.zeros(6,512,512,3, device=self.device)

        layers = [
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            UpSample(), # 128*256

            torch.nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            UpSample(), # 256*512

            torch.nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            UpSample(), # 512*1024

            torch.nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            torch.nn.Conv2d(32, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            UpSample(), # 1024*2048

            torch.nn.Conv2d(32, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            torch.nn.Conv2d(32, 3, 1, padding=0, bias=True),
        ]

        self.net = nn.Sequential(*layers)
        
        # try to load from file
        if path is not None:
            self.load(path)   

    def load(self, path):
        # load latlong env map from file
        image = imageio.imread(path)
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255

        self.image = torch.from_numpy(image).to(self.device) * self.scale
        cubemap = latlong_to_cubemap(self.image, [self.max_res, self.max_res], self.device)

        self.base.data = cubemap

    def build_base(self):
        self.image = self.net(self.init_base).squeeze(0).permute(1,2,0).contiguous()
        self.base = nn.functional.softplus(self.base_train + latlong_to_cubemap(self.image, [self.max_res, self.max_res], self.device))

    def build_mips(self, cutoff=0.99):
        
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)


    def get_mip(self, roughness):
        # map roughness to mip_level (float):
        # roughness: 0 --> self.min_roughness --> self.max_roughness --> 1
        # mip_level: 0 --> 0                  --> M - 2              --> M - 1
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def __call__(self, l, roughness=None):
        import torch.nn.functional as F
        
        # Determine if we are doing Diffuse (no roughness) or Specular (with roughness)
        if roughness is None:
            # Diffuse Path
            tex = self.diffuse[None, ...] if self.diffuse.dim() == 3 else self.diffuse
        else:
            # Specular Path - use the base image (simplified MIP bypass)
            tex = self.image[None, ...] if self.image.dim() == 3 else self.image

        # 1. Prepare Texture [Batch, Channels, H, W]
        if tex.dim() == 3: # Latlong [H, W, C]
            tex_pt = tex.permute(0, 3, 1, 2)
        else: # Cubemap [6, H, W, C]
            tex_pt = tex.permute(0, 3, 1, 2)

        # 2. Prepare Grid [-1, 1]
        # l is [N, 3] or [N, 2]. We need [Batch, 1, N, 2]
        grid = torch.stack([l[..., 0] * 2.0 - 1.0, 1.0 - 2.0 * l[..., 1]], dim=-1)
        grid = grid.reshape(1, 1, -1, 2).expand(tex_pt.shape[0], -1, -1, -1)

        # 3. Sample
        sampled = F.grid_sample(tex_pt, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # 4. Aggregate (Mean across faces if cubemap)
        light = sampled.mean(dim=0).permute(1, 2, 0).reshape(-1, 3)
        
        return light