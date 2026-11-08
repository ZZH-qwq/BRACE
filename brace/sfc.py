"""
Stripe Frequency Consistency (SFC)
"""

import math

import torch
import torch.nn.functional as F


class SFC(torch.nn.Module):
    def __init__(self, max_side=160, min_radius=8, max_radius_ratio=0.45,
                 angle_half_width_deg=10.0, lambda_energy=5.0):
        super().__init__()
        self.max_side = max_side
        self.min_radius = min_radius
        self.max_radius_ratio = max_radius_ratio
        self.angle_half_width_deg = angle_half_width_deg
        self.lambda_energy = lambda_energy

    # ------------------------------------------------------------------
    def _downsample_size(self, h, w):
        scale = min(1.0, float(self.max_side) / float(max(h, w)))
        return max(8, int(round(h * scale))), max(8, int(round(w * scale)))

    @staticmethod
    def _hann2d(h, w, device, dtype):
        wy = torch.hann_window(h, device=device, dtype=dtype)
        wx = torch.hann_window(w, device=device, dtype=dtype)
        return wy[:, None] * wx[None, :]

    def _fft_power(self, img):
        """img: (1, C, H, W) -> power (dh, dw) plus spectrum geometry."""
        _, _, h, w = img.shape
        gray = img.mean(dim=1, keepdim=True)
        dh, dw = self._downsample_size(h, w)
        gray = F.interpolate(gray, size=(dh, dw), mode='bicubic', align_corners=False)
        window = self._hann2d(dh, dw, gray.device, gray.dtype)
        centered = (gray - gray.mean(dim=(-2, -1), keepdim=True)) * window
        spectrum = torch.fft.fftshift(torch.fft.fft2(centered, dim=(-2, -1)), dim=(-2, -1))
        return spectrum.abs().square().squeeze(0).squeeze(0), dh // 2, dw // 2, dh, dw

    def _masks(self, power, cy, cx, dh, dw):
        yy = torch.arange(dh, device=power.device, dtype=power.dtype).view(-1, 1)
        xx = torch.arange(dw, device=power.device, dtype=power.dtype).view(1, -1)
        rr = torch.sqrt((yy - cy).pow(2) + (xx - cx).pow(2))
        max_r = int(min(dh, dw) * self.max_radius_ratio)
        ring = (rr >= self.min_radius) & (rr <= max_r)
        half = ((yy - cy) < 0) | (((yy - cy) == 0) & ((xx - cx) > 0))
        angles = torch.atan2(yy - cy, xx - cx)
        return rr, ring, half, angles, max_r

    def _dominant_angle(self, power, cy, cx, dh, dw):
        _, ring, half, angles, _ = self._masks(power, cy, cx, dh, dw)
        mask = ring & half
        masked = power * mask.to(power.dtype)
        index = masked.reshape(-1).argmax()
        return angles.reshape(-1)[index]

    def _sector(self, power, angle, cy, cx, dh, dw):
        _, ring, half, angles, max_r = self._masks(power, cy, cx, dh, dw)
        diff = torch.abs(angles - angle)
        diff = torch.min(diff, 2.0 * torch.pi - diff)
        sector = ring & half & (diff <= math.radians(self.angle_half_width_deg))
        return sector, ring & half, max_r

    # ------------------------------------------------------------------
    def _profile(self, power, angle, cy, cx, dh, dw, normaliser):
        """Per-radius mean power inside the sector, normalised by the GT sector energy."""
        sector, ring, max_r = self._sector(power, angle, cy, cx, dh, dw)
        rr, _, _, _, _ = self._masks(power, cy, cx, dh, dw)
        ring_energy = (power * ring.to(power.dtype)).sum() + 1e-12

        bins = rr.round().long().clamp(self.min_radius, max_r) - self.min_radius
        num_bins = max_r - self.min_radius + 1
        weights = power * sector.to(power.dtype)
        sums = torch.zeros(num_bins, device=power.device, dtype=power.dtype)
        sums.scatter_add_(0, bins.reshape(-1), weights.reshape(-1))
        counts = torch.zeros(num_bins, device=power.device, dtype=power.dtype)
        counts.scatter_add_(0, bins.reshape(-1), sector.to(power.dtype).reshape(-1))
        profile = (sums / (counts + 1e-12)) / normaliser
        return profile, ring_energy

    def forward(self, pred, gt, lq):
        """pred, gt, lq: (1, 3, H, W) images in [0, 1]."""
        lq_power, cy, cx, dh, dw = self._fft_power(lq)
        angle = self._dominant_angle(lq_power, cy, cx, dh, dw)

        pred_power, pcy, pcx, pdh, pdw = self._fft_power(pred)
        gt_power, gcy, gcx, gdh, gdw = self._fft_power(gt)

        # normalise both spectra by the GT sector energy
        gt_sector, _, _ = self._sector(gt_power, angle, gcy, gcx, gdh, gdw)
        normaliser = (gt_power * gt_sector.to(gt_power.dtype)).sum() + 1e-12
        pred_power = pred_power / normaliser
        gt_power = gt_power / normaliser

        pred_profile, pred_total = self._profile(pred_power, angle, pcy, pcx, pdh, pdw, 1.0)
        gt_profile, gt_total = self._profile(gt_power, angle, gcy, gcx, gdh, gdw, 1.0)

        pred_ratio = pred_profile.sum() / pred_total
        gt_ratio = gt_profile.sum() / gt_total

        ratio_distance = torch.abs(pred_ratio - gt_ratio)
        profile_distance = F.l1_loss(pred_profile, gt_profile)
        gate = torch.max(pred_ratio, gt_ratio).detach()

        total = (ratio_distance + self.lambda_energy * gate * profile_distance) * 100.0
        return {
            'sfc': float(total.item()),
            'ratio': float(ratio_distance.item()) * 100.0,
            'profile': float(profile_distance.item()) * 100.0,
            'pred_ratio': float(pred_ratio.item()) * 100.0,
            'gt_ratio': float(gt_ratio.item()) * 100.0,
            'lq_angle_deg': math.degrees(float(angle.item())),
        }
