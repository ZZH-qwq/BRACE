"""Multi-Scale Sliced Wasserstein Distance (MS-SWD).

Vendored from the official implementation of

    "Multiscale Sliced Wasserstein Distances as Perceptual Color Difference
     Measures", ECCV 2024, https://github.com/real-hjq/MS-SWD
"""

import torch
import torch.nn.functional as F


class GaussianPyramid(torch.nn.Module):
    def __init__(self, num_scale=5):
        super().__init__()
        self.num_scale = num_scale
        self.register_buffer('kernel', self._gauss_kernel(), persistent=False)

    @staticmethod
    def _gauss_kernel():
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]]) / 256.
        return kernel.repeat(3, 1, 1, 1)

    def forward(self, img):
        kernel = self.kernel.to(img.device)
        pyramid = [img]
        for _ in range(self.num_scale - 1):
            img = F.conv2d(F.pad(img, (2, 2, 2, 2), mode='reflect'), kernel, groups=img.shape[1])
            img = img[:, :, ::2, ::2]
            pyramid.append(img)
        return pyramid


def srgb2lab(x):
    """sRGB [0, 1] (N, 3, H, W) -> CIELAB (D65), differentiable."""
    limit = 0.04045
    lin = torch.where(x > limit, torch.pow((torch.clamp(x, min=limit) + 0.055) / 1.055, 2.4), x / 12.92)

    abc = torch.tensor([[10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794],
                        [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
                        [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]],
                       device=x.device, dtype=x.dtype)
    xyz = torch.matmul(abc, lin.view(lin.shape[0], 3, -1)).view_as(lin)
    ref = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]],
                       device=x.device, dtype=x.dtype)
    xyz = xyz / ref

    delta = 6 / 29
    delta_cube = delta ** 3
    factor = 1 / (3 * delta ** 2)
    f_xyz = torch.where(xyz > delta_cube,
                        torch.pow(torch.clamp(xyz, min=delta_cube), 1.0 / 3.0),
                        factor * xyz + 4 / 29)
    L = 116 * f_xyz[:, 1:2] - 16
    a = 500 * (f_xyz[:, 0:1] - f_xyz[:, 1:2])
    b = 200 * (f_xyz[:, 1:2] - f_xyz[:, 2:3])
    return torch.cat((L, a, b), 1)


class MSSWD(torch.nn.Module):
    """MS-SWD distance in CIELAB, lower is better."""

    def __init__(self, num_scale=3, num_proj=64, patch_size=11, stride=1, c=3):
        super().__init__()
        self.num_scale = num_scale
        self.num_proj = num_proj
        self.patch_size = patch_size
        self.stride = stride
        self.c = c
        self.pyramid = GaussianPyramid(num_scale)

    def _projections(self, device, dtype):
        rand = torch.randn(self.num_proj, self.c * self.patch_size ** 2, device=device, dtype=dtype)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)
        return rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def _swd(self, x, y):
        rand = self._projections(x.device, x.dtype)
        pad = self.patch_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        y = F.pad(y, (pad, pad, pad, pad), mode='reflect')
        projx = F.conv2d(x, rand, stride=self.stride).reshape(x.shape[0], self.num_proj, -1)
        projy = F.conv2d(y, rand, stride=self.stride).reshape(y.shape[0], self.num_proj, -1)
        projx, _ = torch.sort(projx, dim=2)
        projy, _ = torch.sort(projy, dim=2)
        return torch.mean(torch.abs(projx - projy), dim=[1, 2])

    def forward(self, x, y):
        """x, y: (N, 3, H, W) sRGB images in [0, 1]."""
        x_pyr, y_pyr = self.pyramid(x), self.pyramid(y)
        total = 0.0
        for n in range(self.num_scale):
            total = total + self._swd(srgb2lab(x_pyr[n]), srgb2lab(y_pyr[n]))
        return total / self.num_scale
