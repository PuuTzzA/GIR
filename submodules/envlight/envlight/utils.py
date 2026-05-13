import os
import numpy as np

import torch
import nvdiffrast.torch as dr


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x


def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        # Wrap both instances of 'res' in int()
        # --- FIXED FORWARD PASS ---
        # If res is a list [16, 16], we take the first value
        r = res[0] if isinstance(res, list) else res
        
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / r, 1.0 - 1.0 / r, int(r), device=device),
            torch.linspace(-1.0 + 1.0 / r, 1.0 - 1.0 / r, int(r), device=device),
            indexing='ij'
        )
        # ---------------------------
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        # --- PYTORCH BYPASS ---
        import torch.nn.functional as F
        tex_pt = latlong_map.unsqueeze(0).permute(0, 3, 1, 2)
        grid = torch.stack([texcoord[..., 0] * 2.0 - 1.0, 1.0 - 2.0 * texcoord[..., 1]], dim=-1).unsqueeze(0)
        cubemap[s, ...] = F.grid_sample(tex_pt, grid, mode='bilinear', padding_mode='border', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        # ----------------------
    return cubemap


def cubemap_to_latlong(cubemap, res, device='cuda'):
    import torch.nn.functional as F

    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
        indexing='ij'
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    # 3D direction vectors [H, W]
    dx = sintheta * sinphi
    dy = costheta
    dz = -sintheta * cosphi

    abs_dx, abs_dy, abs_dz = torch.abs(dx), torch.abs(dy), torch.abs(dz)
    abs_vals = torch.stack([abs_dx, abs_dy, abs_dz], dim=-1)
    max_idx = torch.argmax(abs_vals, dim=-1)  # 0=X, 1=Y, 2=Z

    signs = torch.stack([dx, dy, dz], dim=-1)
    sign_of_max = torch.gather(signs, -1, max_idx.unsqueeze(-1)).squeeze(-1)
    # face_id: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    face_id = max_idx * 2 + (sign_of_max <= 0).long()

    eps = 1e-8
    C = cubemap.shape[-1]
    result = torch.zeros(res[0], res[1], C, dtype=cubemap.dtype, device=device)

    for s in range(6):
        mask = (face_id == s)
        if not mask.any():
            continue

        # Compute UV based on cube_to_dir inverse mapping
        if s == 0:    # +X: dir=(1,-v,-u)*sc  → u=-dz/|dx|, v=-dy/|dx|
            ma = abs_dx[mask].clamp(min=eps); u = -dz[mask] / ma; v = -dy[mask] / ma
        elif s == 1:  # -X: dir=(-1,-v,u)*sc  → u=dz/|dx|, v=-dy/|dx|
            ma = abs_dx[mask].clamp(min=eps); u = dz[mask] / ma; v = -dy[mask] / ma
        elif s == 2:  # +Y: dir=(u,1,v)*sc    → u=dx/|dy|, v=dz/|dy|
            ma = abs_dy[mask].clamp(min=eps); u = dx[mask] / ma; v = dz[mask] / ma
        elif s == 3:  # -Y: dir=(u,-1,-v)*sc  → u=dx/|dy|, v=-dz/|dy|
            ma = abs_dy[mask].clamp(min=eps); u = dx[mask] / ma; v = -dz[mask] / ma
        elif s == 4:  # +Z: dir=(u,-v,1)*sc   → u=dx/|dz|, v=-dy/|dz|
            ma = abs_dz[mask].clamp(min=eps); u = dx[mask] / ma; v = -dy[mask] / ma
        elif s == 5:  # -Z: dir=(-u,-v,-1)*sc → u=-dx/|dz|, v=-dy/|dz|
            ma = abs_dz[mask].clamp(min=eps); u = -dx[mask] / ma; v = -dy[mask] / ma

        # grid_sample grid: [1, 1, N, 2]
        grid = torch.stack([u, v], dim=-1).unsqueeze(0).unsqueeze(0)
        face_tex = cubemap[s].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        sampled = F.grid_sample(face_tex, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # [1, C, 1, N] → [N, C]
        result[mask] = sampled.squeeze(0).squeeze(1).permute(1, 0)

    return result


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            # If res is a list [16, 16] or a tensor [16, 16], grab the first element
            if isinstance(res, (list, tuple, torch.Tensor)) and len(res) > 1:
                r_val = res[0]
            else:
                r_val = res
            
            r_int = int(r_val)
            # --------------------------------------
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / r_int, 1.0 - 1.0 / r_int, r_int, device=dout.device), 
                                    torch.linspace(-1.0 + 1.0 / r_int, 1.0 - 1.0 / r_int, r_int, device=dout.device)
                                   )
            v = safe_normalize(cube_to_dir(s, gx, gy))
            # --- THE ABSOLUTE FINAL BACKWARD BYPASS ---
            import torch.nn.functional as F
            
            # 1. Get the gradient for the current face 's'
            # dout is the incoming gradient dL/dCubemap [6, H, W, 3]
            d_current = dout[s] if dout.dim() == 4 else dout
            
            # 2. Prepare as [1, C, H, W] for PyTorch grid_sample
            # We multiply by 0.25 as per the original original code's logic
            tex_pt = (d_current[None, ...] * 0.25).permute(0, 3, 1, 2)
            
            # 3. Prepare the grid for this specific face: [1, H, W, 2]
            # v contains the UV coordinates for face 's'
            grid = torch.stack([v[..., 0] * 2.0 - 1.0, 1.0 - 2.0 * v[..., 1]], dim=-1)
            if grid.dim() == 3: # [H, W, 2] -> [1, H, W, 2]
                grid = grid.unsqueeze(0)
            elif grid.dim() == 2: # [N, 2] -> [1, N, 1, 2]
                grid = grid.reshape(1, -1, 1, 2)

            # 4. Sample natively (1-to-1 face calculation)
            sampled = F.grid_sample(tex_pt, grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            # 5. Assign to the output (Squeeze out the batch dimension of 1)
            # res: [1, C, H, W] -> [1, H, W, C] -> [H, W, C]
            out[s, ...] = sampled.permute(0, 2, 3, 1).squeeze(0)
            # -------------------------------------------
        return out        