"""Frequency-Aware Banding Prior (FABP)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import kaiming_init


def hann2d(h, w, device, dtype):
    wy = torch.hann_window(h, device=device, dtype=dtype)
    wx = torch.hann_window(w, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


class FABP(nn.Module):
    """Frequency-aware banding prior."""

    def __init__(self, token_dim=64, max_side=160, min_radius=8, max_radius_ratio=0.45, peak_temperature=0.05):
        super().__init__()
        self.token_dim = token_dim
        self.max_side = max_side
        self.min_radius = min_radius
        self.max_radius_ratio = max_radius_ratio
        self.peak_temperature = peak_temperature
        self.mlp = nn.Sequential(
            nn.Linear(16, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        kaiming_init(self.mlp)

    def _masks(self, dh, dw, device):
        yy = torch.arange(dh, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(dw, device=device, dtype=torch.float32).view(1, -1)
        cy, cx = dh // 2, dw // 2
        rr = torch.sqrt((yy - cy).pow(2) + (xx - cx).pow(2))
        ring = (rr >= self.min_radius) & (rr <= min(dh, dw) * self.max_radius_ratio)
        # real signals are conjugate-symmetric: keep one half of the spectrum
        half = ((yy - cy) < 0) | (((yy - cy) == 0) & ((xx - cx) > 0))
        return ring & half, yy, xx, cy, cx

    def _descriptors(self, gray):
        """gray: (b, h, w) -> (b, 8) banding descriptors."""
        b, h, w = gray.shape
        mask, yy, xx, cy, cx = self._masks(h, w, gray.device)

        window = hann2d(h, w, gray.device, gray.dtype)
        centered = (gray - gray.mean(dim=(-2, -1), keepdim=True)) * window

        spectrum = torch.fft.fftshift(torch.fft.fft2(centered, dim=(-2, -1)), dim=(-2, -1))
        power = spectrum.abs().square() * mask.to(gray.dtype)  # (b, h, w)

        flat = power.reshape(b, -1)
        total = flat.sum(dim=-1) + 1e-12

        # deterministic soft-argmax around the dominant peak
        peak = flat.amax(dim=-1, keepdim=True).detach()
        logits = flat / (self.peak_temperature * peak + 1e-8)
        weights = torch.softmax(logits, dim=-1).reshape(b, h, w)

        px = (weights * xx).sum(dim=(-2, -1))
        py = (weights * yy).sum(dim=(-2, -1))
        peak_energy = (weights * power).sum(dim=(-2, -1))

        fx = (px - cx) / float(w)
        fy = (py - cy) / float(h)
        freq = torch.sqrt(fx * fx + fy * fy)

        theta = torch.atan2(fy, fx) + torch.pi * 0.5
        energy_ratio = peak_energy / total

        # stripe phase: projection of the (windowed) signal on the dominant wave
        arg = 2.0 * torch.pi * (fx[:, None, None] * xx + fy[:, None, None] * yy)
        real = (centered * torch.cos(arg)).sum(dim=(-2, -1))
        imag = (centered * torch.sin(arg)).sum(dim=(-2, -1))
        phase = torch.atan2(imag, real)

        return torch.stack(
            [
                freq,
                torch.cos(2.0 * theta),
                torch.sin(2.0 * theta),
                energy_ratio,
                torch.log1p(peak_energy),
                torch.log1p(total),
                torch.cos(phase),
                torch.sin(phase),
            ],
            dim=-1,
        )

    def forward(self, frames):
        """frames: (n, t, 4, h, w) exposure-normalised RAW frames."""
        n, t, _, h, w = frames.shape
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward(frames.float())

    def _forward(self, frames):
        n, t, _, h, w = frames.shape
        if max(h, w) > self.max_side:
            scale = self.max_side / float(max(h, w))
            h, w = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
            frames = F.interpolate(frames.flatten(0, 1), size=(h, w), mode="bicubic", align_corners=False)
        else:
            frames = frames.flatten(0, 1)

        gray = frames.mean(dim=1)  # (n*t, h, w)
        stats = self._descriptors(gray).view(n, t, 8)  # (n, t, 8)

        weights = stats[..., 3].clamp(min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        mean = (stats * weights[..., None]).sum(dim=1)
        var = ((stats - mean[:, None, :]).square() * weights[..., None]).sum(dim=1)
        std = torch.sqrt(var + 1e-6)

        token = self.mlp(torch.cat([mean, std], dim=-1))
        return {"token": token, "stats": stats, "weights": weights}
