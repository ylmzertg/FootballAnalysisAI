from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class TeamClassifierConfig:
    n_clusters: int = 2
    min_bbox_height: int = 28
    min_bbox_width: int = 10
    max_samples_per_track: int = 12
    min_samples_per_track: int = 2
    random_state: int = 7
    pca_components: int = 8
    torso_y0: float = 0.10
    torso_y1: float = 0.58
    torso_x_margin: float = 0.12


class TeamColorClusterer:
    """
    Lightweight offline two-team classifier.

    Feature:
      - torso HSV H/S histogram
      - LAB mean/std
      - HSV mean/std
      - dominant/median BGR colour

    Fit:
      - track-level median prototype
      - StandardScaler
      - PCA
      - KMeans(n_clusters=2)

    This deliberately excludes referees and goalkeepers from team fitting.
    Goalkeepers can be assigned later from spatial proximity to classified players.
    """

    def __init__(self, config: Optional[TeamClassifierConfig] = None):
        self.config = config or TeamClassifierConfig()

        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None

        self.track_features: dict[int, list[np.ndarray]] = defaultdict(list)
        self.track_colors: dict[int, list[np.ndarray]] = defaultdict(list)

        self.track_team: dict[int, int] = {}
        self.team_colors_bgr: dict[int, tuple[int, int, int]] = {}

        self.is_fitted = False

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def crop_torso(self, frame: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
        h_img, w_img = frame.shape[:2]

        x1, y1, x2, y2 = map(float, bbox_xyxy)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        x1 = x1 + bw * self.config.torso_x_margin
        x2 = x2 - bw * self.config.torso_x_margin
        y1 = y1 + bh * self.config.torso_y0
        y2 = y1 + bh * (self.config.torso_y1 - self.config.torso_y0)

        ix1 = self._clamp(int(round(x1)), 0, w_img - 1)
        ix2 = self._clamp(int(round(x2)), ix1 + 1, w_img)
        iy1 = self._clamp(int(round(y1)), 0, h_img - 1)
        iy2 = self._clamp(int(round(y2)), iy1 + 1, h_img)

        return frame[iy1:iy2, ix1:ix2]

    @staticmethod
    def _valid_colour_mask(hsv: np.ndarray) -> np.ndarray:
        """
        Reduce grass/background influence while keeping black/white kits.

        Strong green pixels are removed. Very dark noise is also ignored.
        """
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        strong_green = (h >= 28) & (h <= 95) & (s >= 55) & (v >= 35)
        too_dark = v < 20

        return (~strong_green & ~too_dark).astype(np.uint8) * 255

    def extract_feature(self, crop: np.ndarray):
        if crop is None or crop.size == 0:
            return None

        if crop.shape[0] < 5 or crop.shape[1] < 4:
            return None

        resized = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        mask = self._valid_colour_mask(hsv)

        if cv2.countNonZero(mask) < 60:
            mask = np.full(resized.shape[:2], 255, dtype=np.uint8)

        # H/S joint histogram: 12 x 6 = 72 dimensions
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            mask,
            [12, 6],
            [0, 180, 0, 256],
        ).flatten().astype(np.float32)

        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist /= hist_sum

        selected = mask > 0

        hsv_pixels = hsv[selected].astype(np.float32)
        lab_pixels = lab[selected].astype(np.float32)
        bgr_pixels = resized[selected].astype(np.float32)

        if len(hsv_pixels) == 0:
            return None

        hsv_stats = np.concatenate([
            hsv_pixels.mean(axis=0),
            hsv_pixels.std(axis=0),
        ])
        lab_stats = np.concatenate([
            lab_pixels.mean(axis=0),
            lab_pixels.std(axis=0),
        ])

        # Robust representative kit colour.
        representative_bgr = np.median(bgr_pixels, axis=0)

        # Normalize stats to roughly 0..1 scale.
        hsv_stats = hsv_stats / np.array(
            [180, 255, 255, 180, 255, 255],
            dtype=np.float32,
        )
        lab_stats = lab_stats / 255.0
        color_norm = representative_bgr / 255.0

        feature = np.concatenate([
            hist,
            hsv_stats.astype(np.float32),
            lab_stats.astype(np.float32),
            color_norm.astype(np.float32),
        ]).astype(np.float32)

        norm = float(np.linalg.norm(feature))
        if norm > 1e-8:
            feature /= norm

        return feature, representative_bgr

    def add_sample(
        self,
        track_id: int,
        crop: np.ndarray,
    ) -> bool:
        if len(self.track_features[track_id]) >= self.config.max_samples_per_track:
            return False

        extracted = self.extract_feature(crop)
        if extracted is None:
            return False

        feature, color = extracted

        self.track_features[track_id].append(feature)
        self.track_colors[track_id].append(color)

        return True

    def fit(self) -> None:
        track_ids = []
        prototypes = []
        prototype_colors = []

        for track_id, samples in sorted(self.track_features.items()):
            if len(samples) < self.config.min_samples_per_track:
                continue

            prototype = np.median(
                np.stack(samples, axis=0),
                axis=0,
            )
            color = np.median(
                np.stack(self.track_colors[track_id], axis=0),
                axis=0,
            )

            track_ids.append(track_id)
            prototypes.append(prototype)
            prototype_colors.append(color)

        if len(prototypes) < self.config.n_clusters * 2:
            raise RuntimeError(
                "Not enough stable player tracks to fit two teams. "
                f"Usable tracks: {len(prototypes)}"
            )

        X = np.stack(prototypes, axis=0)

        Xs = self.scaler.fit_transform(X)

        n_components = min(
            self.config.pca_components,
            Xs.shape[0] - 1,
            Xs.shape[1],
        )
        n_components = max(2, n_components)

        self.pca = PCA(
            n_components=n_components,
            random_state=self.config.random_state,
        )
        Xp = self.pca.fit_transform(Xs)

        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=30,
            random_state=self.config.random_state,
        )

        raw_labels = self.kmeans.fit_predict(Xp)

        # Determine raw representative team colours.
        raw_team_colors = {}

        for raw_team in range(self.config.n_clusters):
            colors = [
                prototype_colors[i]
                for i, lbl in enumerate(raw_labels)
                if int(lbl) == raw_team
            ]

            if colors:
                raw_team_colors[raw_team] = np.median(
                    np.stack(colors, axis=0),
                    axis=0,
                )
            else:
                raw_team_colors[raw_team] = np.array([128, 128, 128])

        # Remap KMeans' arbitrary labels to deterministic TEAM_A / TEAM_B.
        # Sort by HSV representative colour, then brightness.
        sortable = []

        for raw_team, bgr in raw_team_colors.items():
            sample = np.uint8([[np.clip(bgr, 0, 255)]])
            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
            key = (
                int(hsv[0]),
                int(hsv[1]),
                int(hsv[2]),
            )
            sortable.append((key, raw_team))

        sortable.sort()
        remap = {
            raw_team: stable_team
            for stable_team, (_key, raw_team) in enumerate(sortable)
        }

        self.track_team.clear()

        for track_id, raw_label in zip(track_ids, raw_labels):
            self.track_team[track_id] = remap[int(raw_label)]

        self.team_colors_bgr = {}

        for raw_team, bgr in raw_team_colors.items():
            stable_team = remap[raw_team]
            self.team_colors_bgr[stable_team] = tuple(
                int(round(x))
                for x in np.clip(bgr, 0, 255)
            )

        self.is_fitted = True

    def predict_crop(self, crop: np.ndarray) -> int | None:
        if not self.is_fitted or self.pca is None or self.kmeans is None:
            return None

        extracted = self.extract_feature(crop)
        if extracted is None:
            return None

        feature, _color = extracted

        X = feature.reshape(1, -1)
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)

        raw_label = int(self.kmeans.predict(Xp)[0])

        # Infer remap by nearest known stable-track prototype vote.
        # Easier and robust: compare raw cluster centre to stable tracks.
        raw_to_stable = {}

        for stable_track, stable_team in self.track_team.items():
            samples = self.track_features.get(stable_track)
            if not samples:
                continue

            prototype = np.median(np.stack(samples, axis=0), axis=0)
            proto_p = self.pca.transform(
                self.scaler.transform(prototype.reshape(1, -1))
            )
            proto_raw = int(self.kmeans.predict(proto_p)[0])

            raw_to_stable.setdefault(proto_raw, stable_team)

        return raw_to_stable.get(raw_label)

    def get_track_team(self, track_id: int) -> int | None:
        return self.track_team.get(track_id)
