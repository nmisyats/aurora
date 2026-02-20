from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import torch

from aurora.camera import Camera
from aurora.bbox import BBox
import aurora.geodesy as geo
from aurora.data import RadarPointCloud


@dataclass
class RayBatch:
    ro: torch.Tensor
    rd: torch.Tensor
    tn: torch.Tensor
    tf: torch.Tensor
    g_ref: torch.Tensor
    wl: Optional[str]

    def __len__(self):
        return len(self.ro)

    @property
    def device(self):
        return self.ro.device

    def __iter__(self):
        return iter((self.ro, self.rd, self.tn, self.tf, self.g_ref, self.wl))

    def as_tuple(self):
        return self.ro, self.rd, self.tn, self.tf, self.g_ref, self.wl


@dataclass
class _CameraRayData:
    ro: torch.Tensor
    rd: torch.Tensor
    tn: torch.Tensor
    tf: torch.Tensor
    bbox_mask: torch.Tensor


@dataclass
class _RayPool:
    ro: torch.Tensor
    rd: torch.Tensor
    tn: torch.Tensor
    tf: torch.Tensor
    g_ref: torch.Tensor
    sample_count: int = 0

    @property
    def num_rays(self):
        return len(self.ro)


class RayDataset:
    """
    Pooled ray dataset for multi-camera, multi-wavelength data.

    The pool is refreshed independently per wavelength. On refresh, we sample a subset
    of images for that wavelength and flatten all valid rays into one tensor pool.
    Batch sampling then draws random rays directly from the pooled tensors.
    """

    def __init__(
        self,
        cameras: List[Camera],
        bbox: BBox,
        device: Optional[torch.device] = None,
        img_pool_size_per_wl: Optional[int] = 8,
        img_pool_duration: Optional[int] = 256,
    ):
        if len(cameras) == 0:
            raise ValueError("Empty camera list.")
        if img_pool_size_per_wl is not None and img_pool_size_per_wl <= 0:
            raise ValueError("img_pool_size_per_wl must be > 0 or None.")
        if img_pool_duration is not None and img_pool_duration <= 0:
            raise ValueError("img_pool_duration must be > 0 or None.")

        self.cameras = cameras
        self.device = device if device is not None else bbox.device
        self.img_pool_size_per_wl = img_pool_size_per_wl
        self.img_pool_duration = img_pool_duration

        self._ray_data_by_wl: Dict[Optional[str], Dict[int, _CameraRayData]] = {}
        self._image_refs_by_wl: Dict[Optional[str], List[Tuple[int, object]]] = {}
        self._pool_by_wl: Dict[Optional[str], _RayPool] = {}

        bbox = bbox.to(self.device)

        for cam_idx, cam in enumerate(self.cameras):
            wl = cam.wavelength

            ro_ecef, rd_ecef = cam.create_rays_ecef(self.device)
            ro = bbox.frame.from_ecef(ro_ecef, is_point=True)
            rd = bbox.frame.from_ecef(rd_ecef, is_point=False)
            tn, tf = bbox.intersection(ro, rd)
            bbox_mask = ~(torch.isnan(tn) | torch.isnan(tf))

            wl_ray_data = self._ray_data_by_wl.setdefault(wl, {})
            wl_ray_data[cam_idx] = _CameraRayData(
                ro=ro[bbox_mask].contiguous(),
                rd=rd[bbox_mask].contiguous(),
                tn=tn[bbox_mask].contiguous(),
                tf=tf[bbox_mask].contiguous(),
                bbox_mask=bbox_mask,
            )

            wl_images = self._image_refs_by_wl.setdefault(wl, [])
            for img in cam.images:
                wl_images.append((cam_idx, img))

        for wl, refs in self._image_refs_by_wl.items():
            if len(refs) == 0:
                raise ValueError(f"No images were found for wavelength '{wl}'.")

    @property
    def wavelengths(self):
        return tuple(self._ray_data_by_wl.keys())

    def _resolve_wavelength(self, wavelength: Optional[str]):
        if wavelength is not None:
            if wavelength not in self._ray_data_by_wl:
                raise ValueError(
                    f"Unknown wavelength '{wavelength}'. Available: {list(self.wavelengths)}"
                )
            return wavelength

        if len(self._ray_data_by_wl) != 1:
            raise ValueError(
                "wavelength must be provided when multiple wavelengths are loaded."
            )
        return next(iter(self._ray_data_by_wl.keys()))

    def _pick_images_for_pool(self, wavelength: Optional[str]):
        refs = list(self._image_refs_by_wl[wavelength])
        if self.img_pool_size_per_wl is None or self.img_pool_size_per_wl >= len(refs):
            random.shuffle(refs)
            return refs
        return random.sample(refs, self.img_pool_size_per_wl)

    def _refresh_pool(self, wavelength: Optional[str]):
        image_refs = self._pick_images_for_pool(wavelength)
        wl_ray_data = self._ray_data_by_wl[wavelength]

        ro_chunks = []
        rd_chunks = []
        tn_chunks = []
        tf_chunks = []
        g_chunks = []

        for cam_idx, img in image_refs:
            cam_data = wl_ray_data[cam_idx]
            g_ref = img.load_image().flatten().to(self.device)

            if g_ref.numel() != cam_data.bbox_mask.numel():
                raise ValueError(
                    f"Image size mismatch for camera index {cam_idx} at wavelength '{wavelength}'. "
                    f"Expected {cam_data.bbox_mask.numel()} pixels, got {g_ref.numel()}."
                )

            g_ref = g_ref[cam_data.bbox_mask].contiguous()
            if g_ref.numel() == 0:
                continue

            ro_chunks.append(cam_data.ro)
            rd_chunks.append(cam_data.rd)
            tn_chunks.append(cam_data.tn)
            tf_chunks.append(cam_data.tf)
            g_chunks.append(g_ref)

        if len(g_chunks) == 0:
            raise ValueError(
                f"Pooled zero rays for wavelength '{wavelength}'. Check bbox coverage and images."
            )

        self._pool_by_wl[wavelength] = _RayPool(
            ro=torch.cat(ro_chunks, dim=0),
            rd=torch.cat(rd_chunks, dim=0),
            tn=torch.cat(tn_chunks, dim=0),
            tf=torch.cat(tf_chunks, dim=0),
            g_ref=torch.cat(g_chunks, dim=0),
            sample_count=0,
        )

    def _pool_needs_refresh(self, wavelength: Optional[str]):
        pool = self._pool_by_wl.get(wavelength)
        if pool is None:
            return True
        if self.img_pool_duration is None:
            return False
        if self.img_pool_size_per_wl is None:
            return False
        img_refs = self._image_refs_by_wl[wavelength]
        if len(img_refs) <= self.img_pool_size_per_wl:
            return False
        return pool.sample_count >= self.img_pool_duration

    def sample_batch(self, batch_size: int, wavelength: Optional[str] = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        wl = self._resolve_wavelength(wavelength)

        if self._pool_needs_refresh(wl):
            self._refresh_pool(wl)

        pool = self._pool_by_wl[wl]
        idxs = torch.randint(0, pool.num_rays, (batch_size,), device=self.device)

        if self.img_pool_duration is not None:
            pool.sample_count += 1

        return RayBatch(
            ro=pool.ro[idxs],
            rd=pool.rd[idxs],
            tn=pool.tn[idxs],
            tf=pool.tf[idxs],
            g_ref=pool.g_ref[idxs],
            wl=wl,
        )


@dataclass
class RadarBatch:
    p: torch.Tensor
    d_ref: torch.Tensor

    def __len__(self):
        return len(self.p)

    @property
    def device(self):
        return self.p.device

    def __iter__(self):
        return iter((self.p, self.d_ref))

    def as_tuple(self):
        return self.p, self.d_ref


class RadarDataset:
    def __init__(self, data: RadarPointCloud, bbox: BBox, device: Optional[torch.device] = None):
        if len(data.latitudes) == 0:
            raise ValueError("Empty radar cloud.")

        self.device = device if device is not None else bbox.device

        bbox = bbox.to(self.device)

        lat = data.latitudes.to(device)
        lon = data.longitudes.to(device)
        h = data.altitudes.to(device)
        d = data.densities.to(device)

        p_ecef = geo.geodetic_to_ecef(lat, lon, h)
        p = bbox.frame.from_ecef(p_ecef, is_point=True)

        inside_mask = bbox.contains(p)

        self.p = p[inside_mask].contiguous()
        self.d_ref = d[inside_mask].contiguous()
        if len(self.p) == 0:
            raise ValueError("No radar points inside the bounding box.")

    def sample_batch(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        idxs = torch.randint(0, len(self.p), (batch_size,), device=self.device)
        return RadarBatch(self.p[idxs], self.d_ref[idxs])
