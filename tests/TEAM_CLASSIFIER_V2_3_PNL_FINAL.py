from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import math

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - handled at runtime with a clear error
    KMeans = None


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN"
REFEREE = "REFEREE"
GOALKEEPER = "GOALKEEPER"
PLAYER = "PLAYER"


@dataclass
class TeamClassifierV2Config:
    """Runtime knobs tuned for broadcast football and modest GPUs such as GTX 1050."""

    # Feature fusion
    embedding_weight: float = 0.55
    colour_weight: float = 0.45
    embedding_stride: int = 5
    embedding_batch_size: int = 24
    use_deep_embedding: bool = True
    embedding_device: str = "auto"

    # Bootstrap / prototype learning
    bootstrap_min_samples: int = 30
    bootstrap_max_samples: int = 220
    bootstrap_samples_per_track: int = 4
    min_crop_width: int = 12
    min_crop_height: int = 28
    prototype_momentum: float = 0.04
    min_prototype_update_confidence: float = 0.64

    # Team decision
    min_team_similarity: float = 0.18
    min_team_margin: float = 0.035
    reliable_team_confidence: float = 0.58

    # Temporal voting
    history_size: int = 20
    temporal_decay: float = 0.90
    min_votes_before_stable: int = 3
    stale_track_frames: int = 180

    # ID-switch / appearance-change awareness
    appearance_switch_cosine_distance: float = 0.38
    colour_switch_cosine_distance: float = 0.34
    hard_switch_cosine_distance: float = 0.55
    hard_colour_switch_cosine_distance: float = 0.30
    max_bbox_jump_factor: float = 6.0
    soft_reset_keep_votes: int = 2

    # Role correction
    # Automatic referee detection must be deliberately conservative.
    # Generic ImageNet embeddings for football players are naturally similar;
    # using referee-prototype similarity alone can therefore spread one false
    # referee label to an entire team. Require repeated *team + colour* outlier
    # evidence before using the referee prototype.
    role_outlier_similarity: float = 0.34
    role_outlier_colour_similarity: float = 0.46
    referee_min_frames: int = 5
    referee_similarity: float = 0.86

    # Detector role hints are noisy on dark kits. A REFEREE hint is therefore
    # evidence, not an immediate hard label. The track must become referee-dominant
    # over time and remain visually distinct from the two team prototypes.
    referee_hint_min_frames: int = 8
    referee_hint_min_total_frames: int = 10
    referee_hint_min_ratio: float = 0.85
    referee_hint_min_confidence: float = 0.55
    referee_hint_min_novelty: float = 0.20

    # Automatic referee promotion is only used after a trusted referee prototype
    # exists, mainly to recover from a ByteTrack ID switch. It is intentionally
    # sparse and conservative.
    referee_max_auto_candidates: int = 1
    referee_min_novelty: float = 0.32
    referee_hard_novelty: float = 0.46
    referee_candidate_prototype_guard_similarity: float = 0.86
    referee_prototype_momentum: float = 0.06
    referee_team_prototype_guard_votes: int = 2
    goalkeeper_goal_line_distance_m: float = 19.0
    goalkeeper_neighbor_radius_m: float = 28.0
    goalkeeper_min_neighbor_votes: int = 2

    # PnLCalib spatial role gating (V2.3)
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    pitch_role_margin_m: float = 1.5

    # Once a track is confirmed to be physically on the pitch, intermittent
    # detector referee hints can be trusted with a much lower purity requirement.
    # Visual novelty vs both teams remains mandatory, which rejects dark-kit
    # players that the detector sporadically calls "referee".
    referee_on_pitch_hint_min_frames: int = 3
    referee_on_pitch_min_total_frames: int = 10
    referee_on_pitch_min_novelty: float = 0.20


@dataclass
class DetectionObservation:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float = 1.0
    pitch_xy: Optional[Tuple[float, float]] = None
    role_hint: Optional[str] = None


@dataclass
class TeamAssignment:
    track_id: int
    team: str
    role: str
    confidence: float
    raw_team: str
    raw_confidence: float
    id_switch_suspected: bool = False
    reason: str = ""


@dataclass
class _Vote:
    team: str
    confidence: float
    frame_index: int


@dataclass
class _TrackState:
    votes: Deque[_Vote]
    embedding_proto: Optional[np.ndarray] = None
    colour_proto: Optional[np.ndarray] = None
    feature_proto: Optional[np.ndarray] = None
    last_embedding: Optional[np.ndarray] = None
    last_embedding_frame: int = -10_000
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_frame: int = -1
    seen_frames: int = 0
    referee_votes: int = 0
    referee_hint_frames: int = 0
    non_referee_hint_frames: int = 0
    referee_hint_streak: int = 0
    referee_on_pitch_hint_frames: int = 0
    on_pitch_seen_frames: int = 0
    referee_trusted: bool = False
    goalkeeper_votes: int = 0


class _AppearanceEncoder:
    """Lazy MobileNetV3-small encoder with a dependency-safe fallback."""

    def __init__(self, config: TeamClassifierV2Config):
        self.config = config
        self.available = False
        self.device = "cpu"
        self.model = None
        self.transform = None
        self._torch = None

        if not config.use_deep_embedding:
            return

        try:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            if config.embedding_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.embedding_device

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            # Features + global average pooling. The classifier is intentionally removed.
            self.model = model.features.eval().to(self.device)
            self.transform = weights.transforms()
            self._torch = torch
            self.available = True
        except Exception:
            # Colour + lightweight visual descriptor remain usable. We deliberately
            # avoid making team classification fail because torchvision/model weights
            # are unavailable on a machine.
            self.available = False

    def encode(self, crops_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not crops_bgr:
            return []
        if not self.available:
            return [self._fallback_descriptor(crop) for crop in crops_bgr]

        torch = self._torch
        assert torch is not None and self.model is not None and self.transform is not None

        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(torch.from_numpy(rgb).permute(2, 0, 1)))

        out: List[np.ndarray] = []
        batch_size = max(1, self.config.embedding_batch_size)
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size]).to(self.device)
                feats = self.model(batch)
                feats = feats.mean(dim=(-2, -1))
                feats = torch.nn.functional.normalize(feats, dim=1)
                out.extend(feats.detach().cpu().numpy().astype(np.float32))
        return out

    @staticmethod
    def _fallback_descriptor(crop_bgr: np.ndarray) -> np.ndarray:
        """Cheap appearance descriptor used when torchvision is unavailable.

        It is intentionally not just a colour histogram: grayscale structure and
        edge layout preserve coarse shirt/short texture information.
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (12, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        descriptor = np.concatenate([
            cv2.resize(small, (6, 12), interpolation=cv2.INTER_AREA).ravel(),
            cv2.resize(mag, (6, 12), interpolation=cv2.INTER_AREA).ravel(),
        ]).astype(np.float32)
        return _l2_normalize(descriptor)


class TeamClassifierV2:
    """Online team / referee / goalkeeper classifier for tracked football players.

    Design goals:
    - fused visual embedding + colour feature
    - rolling temporal voting instead of a permanent fixed-team lock
    - track appearance prototypes to notice ByteTrack ID switches
    - referee and goalkeeper correction using appearance + calibrated pitch position
    - deep embedding refresh throttled for GTX 1050-class hardware

    Typical integration:
        classifier = TeamClassifierV2()
        assignments = classifier.classify_frame(frame_bgr, observations, frame_index)
    """

    def __init__(self, config: Optional[TeamClassifierV2Config] = None):
        self.config = config or TeamClassifierV2Config()
        self.encoder = _AppearanceEncoder(self.config)
        self.track_states: Dict[int, _TrackState] = {}
        self.team_prototypes: Dict[str, np.ndarray] = {}
        self.team_colour_prototypes: Dict[str, np.ndarray] = {}
        self.referee_prototype: Optional[np.ndarray] = None
        self._seen_referee_detector_hint: bool = False

        self._bootstrap_features: List[np.ndarray] = []
        self._bootstrap_track_counts: MutableMapping[int, int] = defaultdict(int)
        self._bootstrapped = False

    @property
    def is_ready(self) -> bool:
        return self._bootstrapped and TEAM_A in self.team_prototypes and TEAM_B in self.team_prototypes

    @property
    def embedding_backend(self) -> str:
        if self.encoder.available:
            return f"mobilenet_v3_small:{self.encoder.device}"
        return "fallback_visual_descriptor:cpu"

    def reset(self) -> None:
        self.track_states.clear()
        self.team_prototypes.clear()
        self.team_colour_prototypes.clear()
        self.referee_prototype = None
        self._seen_referee_detector_hint = False
        self._bootstrap_features.clear()
        self._bootstrap_track_counts.clear()
        self._bootstrapped = False

    def classify_frame(
        self,
        frame_bgr: np.ndarray,
        observations: Sequence[DetectionObservation | Mapping[str, object]],
        frame_index: int,
    ) -> List[TeamAssignment]:
        obs = [self._coerce_observation(item) for item in observations]
        self._expire_stale_tracks(frame_index)

        valid: List[Tuple[DetectionObservation, np.ndarray]] = []
        for item in obs:
            crop = self._extract_player_crop(frame_bgr, item.bbox_xyxy)
            if crop is not None:
                valid.append((item, crop))

        if not valid:
            return []

        # Colour is cheap and recomputed every frame. Deep embeddings are cached
        # per track and refreshed at a controlled cadence, or immediately after a
        # strong colour change suggests a possible ID switch.
        colours: Dict[int, np.ndarray] = {}
        embeds: Dict[int, np.ndarray] = {}
        to_encode: List[np.ndarray] = []
        encode_track_ids: List[int] = []

        for item, crop in valid:
            state = self._state(item.track_id)
            colour = self._colour_feature(crop)
            colours[item.track_id] = colour

            colour_changed = (
                state.colour_proto is not None
                and _cosine_distance(colour, state.colour_proto) >= self.config.colour_switch_cosine_distance
            )
            embedding_due = (
                state.last_embedding is None
                or frame_index - state.last_embedding_frame >= max(1, self.config.embedding_stride)
                or colour_changed
            )
            if embedding_due:
                to_encode.append(crop)
                encode_track_ids.append(item.track_id)
            else:
                embeds[item.track_id] = state.last_embedding  # type: ignore[assignment]

        fresh_embeddings = self.encoder.encode(to_encode)
        for track_id, embedding in zip(encode_track_ids, fresh_embeddings):
            state = self._state(track_id)
            state.last_embedding = embedding
            state.last_embedding_frame = frame_index
            embeds[track_id] = embedding

        fused: Dict[int, np.ndarray] = {
            item.track_id: self._fuse_features(embeds[item.track_id], colours[item.track_id])
            for item, _ in valid
        }

        self._collect_bootstrap(valid, fused)
        if not self.is_ready:
            self._try_bootstrap()

        preliminary: Dict[int, TeamAssignment] = {}
        for item, _ in valid:
            track_id = item.track_id
            state = self._state(track_id)
            embedding = embeds[track_id]
            colour = colours[track_id]
            feature = fused[track_id]

            switch_suspected, switch_strength, switch_reason = self._detect_id_switch(
                state, item.bbox_xyxy, embedding, colour, frame_index
            )
            if switch_suspected:
                self._handle_possible_switch(state, switch_strength)

            raw_team, raw_conf, raw_reason = self._classify_team(feature)
            if raw_team in (TEAM_A, TEAM_B) and raw_conf > 0:
                state.votes.append(_Vote(raw_team, raw_conf, frame_index))

            voted_team, voted_conf = self._temporal_team(state, raw_team, raw_conf, frame_index)
            reason_parts = [raw_reason]
            if switch_reason:
                reason_parts.append(switch_reason)
            if voted_team != raw_team and voted_team != UNKNOWN:
                reason_parts.append("temporal_vote")

            preliminary[track_id] = TeamAssignment(
                track_id=track_id,
                team=voted_team,
                role=PLAYER,
                confidence=voted_conf,
                raw_team=raw_team,
                raw_confidence=raw_conf,
                id_switch_suspected=switch_suspected,
                reason=";".join(part for part in reason_parts if part),
            )

            self._update_track_appearance(state, feature, embedding, colour, switch_suspected)
            state.last_bbox = item.bbox_xyxy
            state.last_frame = frame_index
            state.seen_frames += 1

        self._correct_roles(valid, fused, colours, preliminary)
        self._update_global_prototypes(valid, fused, colours, preliminary)
        return [preliminary[item.track_id] for item, _ in valid]

    def debug_state(self) -> Dict[str, object]:
        return {
            "ready": self.is_ready,
            "embedding_backend": self.embedding_backend,
            "tracks": len(self.track_states),
            "bootstrap_samples": len(self._bootstrap_features),
            "team_prototypes": sorted(self.team_prototypes.keys()),
            "team_colour_prototypes": sorted(self.team_colour_prototypes.keys()),
            "has_referee_prototype": self.referee_prototype is not None,
        }

    def _state(self, track_id: int) -> _TrackState:
        if track_id not in self.track_states:
            self.track_states[track_id] = _TrackState(votes=deque(maxlen=self.config.history_size))
        return self.track_states[track_id]

    @staticmethod
    def _coerce_observation(item: DetectionObservation | Mapping[str, object]) -> DetectionObservation:
        if isinstance(item, DetectionObservation):
            return item
        return DetectionObservation(
            track_id=int(item["track_id"]),
            bbox_xyxy=tuple(float(v) for v in item["bbox_xyxy"]),  # type: ignore[arg-type]
            confidence=float(item.get("confidence", 1.0)),
            pitch_xy=(
                tuple(float(v) for v in item["pitch_xy"])  # type: ignore[arg-type]
                if item.get("pitch_xy") is not None
                else None
            ),
            role_hint=str(item["role_hint"]) if item.get("role_hint") is not None else None,
        )

    def _extract_player_crop(
        self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1i = max(0, min(width - 1, int(round(x1))))
        y1i = max(0, min(height - 1, int(round(y1))))
        x2i = max(1, min(width, int(round(x2))))
        y2i = max(1, min(height, int(round(y2))))
        if x2i - x1i < self.config.min_crop_width or y2i - y1i < self.config.min_crop_height:
            return None

        crop = frame_bgr[y1i:y2i, x1i:x2i]
        # Team identity lives mostly in shirt/shorts. Remove head and lower legs to
        # reduce skin, grass, socks and advertisement contamination.
        h = crop.shape[0]
        top = int(round(h * 0.10))
        bottom = max(top + 1, int(round(h * 0.76)))
        torso = crop[top:bottom]
        if torso.size == 0:
            return crop
        return torso

    @staticmethod
    def _colour_feature(crop_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Remove green-ish pixels (pitch leaking into the player box) and very dark
        # pixels. The mask is intentionally broad because broadcast colour grading
        # varies heavily between sources.
        green = ((h >= 32) & (h <= 92) & (s >= 45))
        useful = (~green) & (v >= 35)
        mask = (useful.astype(np.uint8) * 255)
        if int(mask.sum()) < 255 * 20:
            mask = None

        hist_h = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).ravel()
        hist_s = cv2.calcHist([hsv], [1], mask, [10], [0, 256]).ravel()
        hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).ravel()

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        if mask is None:
            pixels = lab.reshape(-1, 3)
        else:
            pixels = lab[mask.astype(bool)]
        if len(pixels) == 0:
            moments = np.zeros(6, dtype=np.float32)
        else:
            mean = pixels.mean(axis=0)
            std = pixels.std(axis=0)
            moments = np.concatenate([mean, std]).astype(np.float32)

        feature = np.concatenate([hist_h, hist_s, hist_v, moments]).astype(np.float32)
        return _l2_normalize(feature)

    def _fuse_features(self, embedding: np.ndarray, colour: np.ndarray) -> np.ndarray:
        ew = math.sqrt(max(0.0, self.config.embedding_weight))
        cw = math.sqrt(max(0.0, self.config.colour_weight))
        fused = np.concatenate([_l2_normalize(embedding) * ew, _l2_normalize(colour) * cw])
        return _l2_normalize(fused.astype(np.float32))

    def _collect_bootstrap(
        self,
        valid: Sequence[Tuple[DetectionObservation, np.ndarray]],
        fused: Mapping[int, np.ndarray],
    ) -> None:
        if self.is_ready or len(self._bootstrap_features) >= self.config.bootstrap_max_samples:
            return
        for item, _ in valid:
            if self._bootstrap_track_counts[item.track_id] >= self.config.bootstrap_samples_per_track:
                continue
            self._bootstrap_features.append(fused[item.track_id].copy())
            self._bootstrap_track_counts[item.track_id] += 1
            if len(self._bootstrap_features) >= self.config.bootstrap_max_samples:
                break

    def _try_bootstrap(self) -> None:
        if self.is_ready or len(self._bootstrap_features) < self.config.bootstrap_min_samples:
            return
        if KMeans is None:
            raise RuntimeError("scikit-learn is required for TeamClassifierV2 bootstrap clustering")

        matrix = np.vstack(self._bootstrap_features).astype(np.float32)
        km = KMeans(n_clusters=2, n_init=10, random_state=17)
        labels = km.fit_predict(matrix)
        counts = Counter(labels)
        if min(counts.values()) < max(4, self.config.bootstrap_min_samples // 6):
            return

        centers = np.vstack([matrix[labels == idx].mean(axis=0) for idx in range(2)])
        centers = np.vstack([_l2_normalize(row) for row in centers])

        # Team names are anonymous. KMeans' deterministic random_state keeps the
        # mapping stable for the run while avoiding any false home/away semantics.
        self.team_prototypes[TEAM_A] = centers[0]
        self.team_prototypes[TEAM_B] = centers[1]
        self._bootstrapped = True

    def _classify_team(self, feature: np.ndarray) -> Tuple[str, float, str]:
        if not self.is_ready:
            return UNKNOWN, 0.0, "bootstrap_pending"

        sim_a = _cosine_similarity(feature, self.team_prototypes[TEAM_A])
        sim_b = _cosine_similarity(feature, self.team_prototypes[TEAM_B])
        if sim_a >= sim_b:
            team, best, second = TEAM_A, sim_a, sim_b
        else:
            team, best, second = TEAM_B, sim_b, sim_a

        margin = best - second
        if best < self.config.min_team_similarity:
            return UNKNOWN, 0.0, "team_outlier"

        # Confidence combines inter-team separation and absolute prototype fit.
        margin_score = _sigmoid((margin - self.config.min_team_margin) * 10.0)
        fit_score = np.clip((best - self.config.min_team_similarity) / (1.0 - self.config.min_team_similarity), 0.0, 1.0)
        confidence = float(0.70 * margin_score + 0.30 * fit_score)
        return team, confidence, f"prototype:{best:.3f}/{second:.3f}"

    def _temporal_team(
        self,
        state: _TrackState,
        raw_team: str,
        raw_confidence: float,
        frame_index: int,
    ) -> Tuple[str, float]:
        if not state.votes:
            return raw_team, raw_confidence

        scores = {TEAM_A: 0.0, TEAM_B: 0.0}
        total = 0.0
        for vote in reversed(state.votes):
            age = max(0, frame_index - vote.frame_index)
            weight = (self.config.temporal_decay ** age) * max(0.05, vote.confidence)
            scores[vote.team] += weight
            total += weight

        if total <= 0:
            return raw_team, raw_confidence

        best_team = max(scores, key=scores.get)
        best_score = scores[best_team]
        confidence = float(best_score / total)

        if len(state.votes) < self.config.min_votes_before_stable and raw_team != UNKNOWN:
            return raw_team, max(raw_confidence, confidence * 0.8)
        return best_team, confidence

    def _detect_id_switch(
        self,
        state: _TrackState,
        bbox: Tuple[float, float, float, float],
        embedding: np.ndarray,
        colour: np.ndarray,
        frame_index: int,
    ) -> Tuple[bool, float, str]:
        if state.seen_frames < 2 or state.last_frame < 0:
            return False, 0.0, ""

        emb_dist = _cosine_distance(embedding, state.embedding_proto) if state.embedding_proto is not None else 0.0
        col_dist = _cosine_distance(colour, state.colour_proto) if state.colour_proto is not None else 0.0
        jump = self._bbox_jump_factor(state.last_bbox, bbox) if state.last_bbox is not None else 0.0
        gap = frame_index - state.last_frame

        appearance_alarm = (
            emb_dist >= self.config.appearance_switch_cosine_distance
            and col_dist >= self.config.colour_switch_cosine_distance
        )
        hard_alarm = (
            emb_dist >= self.config.hard_switch_cosine_distance
            or col_dist >= self.config.hard_colour_switch_cosine_distance
        )
        teleport_alarm = gap <= 8 and jump >= self.config.max_bbox_jump_factor and (emb_dist > 0.20 or col_dist > 0.20)

        suspected = hard_alarm or appearance_alarm or teleport_alarm
        strength = max(
            emb_dist / max(1e-6, self.config.hard_switch_cosine_distance),
            col_dist / max(1e-6, self.config.colour_switch_cosine_distance),
            jump / max(1e-6, self.config.max_bbox_jump_factor) if gap <= 8 else 0.0,
        )
        if not suspected:
            return False, float(strength), ""
        return True, float(strength), f"id_switch? emb={emb_dist:.2f} col={col_dist:.2f} jump={jump:.1f}"

    def _handle_possible_switch(self, state: _TrackState, strength: float) -> None:
        # A hard appearance break behaves like a new person reusing the same
        # ByteTrack ID. A softer break retains only the newest few votes so the
        # classifier can recover without one-frame flicker.
        if strength >= 1.15:
            state.votes.clear()
            state.embedding_proto = None
            state.colour_proto = None
            state.feature_proto = None
            state.referee_votes = 0
            state.referee_hint_frames = 0
            state.non_referee_hint_frames = 0
            state.referee_hint_streak = 0
            state.referee_trusted = False
            state.goalkeeper_votes = 0
        elif len(state.votes) > self.config.soft_reset_keep_votes:
            newest = list(state.votes)[-self.config.soft_reset_keep_votes :]
            state.votes.clear()
            state.votes.extend(newest)

    def _update_track_appearance(
        self,
        state: _TrackState,
        feature: np.ndarray,
        embedding: np.ndarray,
        colour: np.ndarray,
        switch_suspected: bool,
    ) -> None:
        if switch_suspected and state.feature_proto is not None:
            return
        alpha = 0.18 if state.seen_frames < 3 else 0.06
        state.feature_proto = _ema_unit(state.feature_proto, feature, alpha)
        state.embedding_proto = _ema_unit(state.embedding_proto, embedding, alpha)
        state.colour_proto = _ema_unit(state.colour_proto, colour, alpha)

    def _correct_roles(
        self,
        valid: Sequence[Tuple[DetectionObservation, np.ndarray]],
        fused: Mapping[int, np.ndarray],
        colours: Mapping[int, np.ndarray],
        assignments: MutableMapping[int, TeamAssignment],
    ) -> None:
        """Correct referee / goalkeeper roles with track-level role consensus.

        Detector ``referee`` labels are deliberately treated as *hints*. Broadcast
        football detectors often confuse dark outfield kits with referees, so a
        single hint must never override the team classifier.

        V2.3 rules:
        - PnLCalib ``pitch_xy`` is a spatial role gate: off-pitch referee hints can
          never seed the trusted referee prototype;
        - without calibration, retain the conservative V2.2 sustained detector
          consensus (minimum frames + purity ratio + visual novelty);
        - with an on-pitch coordinate, intermittent referee hints are allowed to
          accumulate across the track, but visual novelty vs both teams remains
          mandatory (this recovers the real match referee while rejecting dark-kit
          players and sideline staff);
        - once trusted, that track alone teaches the referee appearance prototype;
        - goalkeeper hints require calibrated pitch coordinates near a goal line.
        """
        observations = {item.track_id: item for item, _ in valid}
        handled: set[int] = set()

        # Pass 1: accumulate detector role evidence for every visible track.
        for track_id, assignment in assignments.items():
            item = observations[track_id]
            state = self._state(track_id)
            hint = item.role_hint.upper() if item.role_hint else ""

            on_pitch = self._inside_pitch(item.pitch_xy)

            if hint == REFEREE:
                self._seen_referee_detector_hint = True

            if on_pitch:
                state.on_pitch_seen_frames += 1

            if hint == REFEREE and item.confidence >= self.config.referee_hint_min_confidence:
                state.referee_hint_frames += 1
                state.referee_hint_streak += 1
                if on_pitch:
                    state.referee_on_pitch_hint_frames += 1
            else:
                state.non_referee_hint_frames += 1
                state.referee_hint_streak = max(0, state.referee_hint_streak - 1)

        # Pass 2: explicit role hints + calibrated goalkeeper handling.
        for track_id, assignment in assignments.items():
            item = observations[track_id]
            feature = fused[track_id]
            colour = colours[track_id]
            state = self._state(track_id)
            hint = item.role_hint.upper() if item.role_hint else ""

            team_similarity = self._best_team_similarity(feature)
            colour_similarity = self._best_team_colour_similarity(colour)
            colour_novelty = max(0.0, 1.0 - colour_similarity)
            feature_novelty = max(0.0, 1.0 - team_similarity)
            novelty = 0.80 * colour_novelty + 0.20 * feature_novelty

            total_hints = state.referee_hint_frames + state.non_referee_hint_frames
            hint_ratio = (
                state.referee_hint_frames / total_hints
                if total_hints > 0
                else 0.0
            )
            on_pitch = self._inside_pitch(item.pitch_xy)
            off_pitch = item.pitch_xy is not None and not on_pitch

            if hint == REFEREE or state.referee_trusted:
                # Never promote a spatially off-pitch track to match referee.
                # This is the decisive gate for sideline staff / assistant officials.
                if off_pitch:
                    assignment.reason += ";referee_hint_rejected_off_pitch"
                    state.referee_votes = max(0, state.referee_votes - 1)
                    continue

                enough_global_ref_hints = (
                    state.referee_hint_frames >= self.config.referee_hint_min_frames
                    and total_hints >= self.config.referee_hint_min_total_frames
                    and hint_ratio >= self.config.referee_hint_min_ratio
                )

                enough_on_pitch_ref_hints = (
                    on_pitch
                    and state.referee_on_pitch_hint_frames
                    >= self.config.referee_on_pitch_hint_min_frames
                    and state.on_pitch_seen_frames
                    >= self.config.referee_on_pitch_min_total_frames
                )

                enough_ref_hints = (
                    enough_global_ref_hints or enough_on_pitch_ref_hints
                )

                novelty_threshold = (
                    self.config.referee_on_pitch_min_novelty
                    if enough_on_pitch_ref_hints
                    else self.config.referee_hint_min_novelty
                )
                visually_plausible_referee = novelty >= novelty_threshold

                # If we already have a trusted referee prototype, it can rescue a
                # lower-novelty crop, but detector purity is still mandatory.
                prototype_similarity = (
                    _cosine_similarity(feature, self.referee_prototype)
                    if self.referee_prototype is not None
                    else -1.0
                )
                prototype_agrees = (
                    self.referee_prototype is not None
                    and prototype_similarity >= self.config.referee_similarity
                )

                if state.referee_trusted or (
                    enough_ref_hints
                    and (visually_plausible_referee or prototype_agrees)
                ):
                    state.referee_trusted = True
                    self._mark_referee(
                        state,
                        feature,
                        assignment,
                        (
                            f"trusted_role_consensus:"
                            f"ref={state.referee_hint_frames}/"
                            f"{total_hints},ratio={hint_ratio:.2f},"
                            f"onpitch_ref={state.referee_on_pitch_hint_frames}/"
                            f"{state.on_pitch_seen_frames},"
                            f"novelty={novelty:.2f}"
                        ),
                        trusted_hint=True,
                    )
                    handled.add(track_id)
                    continue

                # Reject noisy per-frame REF hints and leave the normal TEAM_A/B
                # decision intact. This is the key V2.2 correction for dark kits.
                if hint == REFEREE:
                    assignment.reason += (
                        f";referee_hint_pending:"
                        f"ref={state.referee_hint_frames}/"
                        f"{total_hints},ratio={hint_ratio:.2f},"
                        f"novelty={novelty:.2f}"
                    )

            if hint == GOALKEEPER:
                team = self._goalkeeper_team(item, observations, assignments)
                if team in (TEAM_A, TEAM_B) and self._near_goal_line(item.pitch_xy):
                    assignment.team = team
                    assignment.role = GOALKEEPER
                    assignment.confidence = max(assignment.confidence, 0.80)
                    assignment.reason += ";goalkeeper_role_hint+pitch"
                    state.goalkeeper_votes += 1
                else:
                    assignment.reason += ";goalkeeper_hint_waiting_for_pitch"
                handled.add(track_id)
                continue

            near_goal = self._near_goal_line(item.pitch_xy)
            if near_goal and (
                colour_similarity < self.config.role_outlier_colour_similarity
                or team_similarity < self.config.role_outlier_similarity
            ):
                team = self._goalkeeper_team(item, observations, assignments)
                if team in (TEAM_A, TEAM_B):
                    assignment.team = team
                    assignment.role = GOALKEEPER
                    assignment.confidence = max(assignment.confidence, 0.72)
                    assignment.reason += ";goalkeeper_pitch_correction"
                    state.goalkeeper_votes += 1
                    state.referee_votes = max(0, state.referee_votes - 1)
                    handled.add(track_id)

        # Pass 3: if the upstream detector has exposed a REFEREE class, never
        # seed a referee from appearance-only outliers. We wait for a spatially
        # valid detector-consensus track above. If no referee class has ever been
        # seen, retain the conservative appearance-only fallback for compatibility
        # with generic person/player detectors.
        if self.referee_prototype is None and self._seen_referee_detector_hint:
            for track_id, assignment in assignments.items():
                if track_id not in handled:
                    self._state(track_id).referee_votes = max(
                        0, self._state(track_id).referee_votes - 1
                    )
            return

        candidates: List[Tuple[float, int, float]] = []
        for track_id, assignment in assignments.items():
            if track_id in handled:
                continue

            item = observations[track_id]
            if self._near_goal_line(item.pitch_xy):
                continue

            feature = fused[track_id]
            colour = colours[track_id]
            team_similarity = self._best_team_similarity(feature)
            colour_similarity = self._best_team_colour_similarity(colour)
            colour_novelty = max(0.0, 1.0 - colour_similarity)
            feature_novelty = max(0.0, 1.0 - team_similarity)
            novelty = 0.80 * colour_novelty + 0.20 * feature_novelty
            ref_similarity = _cosine_similarity(feature, self.referee_prototype)

            if self.referee_prototype is None:
                # Fallback mode only: no upstream referee hints exist anywhere.
                eligible = (
                    novelty >= self.config.referee_hard_novelty
                    and (
                        colour_similarity < self.config.role_outlier_colour_similarity
                        or team_similarity < self.config.role_outlier_similarity
                    )
                )
                score = novelty
            else:
                eligible = (
                    novelty >= self.config.referee_min_novelty
                    and ref_similarity
                    >= self.config.referee_candidate_prototype_guard_similarity
                )
                score = 0.70 * ref_similarity + 0.30 * novelty

            if eligible:
                candidates.append((score, track_id, novelty))

        candidates.sort(key=lambda row: row[0], reverse=True)
        max_candidates = max(0, self.config.referee_max_auto_candidates)
        selected = {row[1]: row for row in candidates[:max_candidates]}

        for track_id, assignment in assignments.items():
            if track_id in handled:
                continue

            state = self._state(track_id)
            candidate = selected.get(track_id)
            if candidate is None:
                state.referee_votes = max(0, state.referee_votes - 1)
                continue

            state.referee_votes += 1
            if state.referee_votes < self.config.referee_min_frames:
                continue

            _, _, novelty = candidate
            self._mark_referee(
                state,
                fused[track_id],
                assignment,
                f"trusted_prototype_recovery:novelty={novelty:.2f}",
                trusted_hint=False,
            )

    def _mark_referee(
        self,
        state: _TrackState,
        feature: np.ndarray,
        assignment: TeamAssignment,
        reason: str,
        trusted_hint: bool = False,
    ) -> None:
        assignment.team = UNKNOWN
        assignment.role = REFEREE
        assignment.confidence = max(assignment.confidence, 0.72)
        assignment.reason += f";referee:{reason}"

        # Only a track that passed detector-consensus + visual-novelty gating may
        # teach the prototype. Automatic recovery tracks never update it.
        if trusted_hint and state.referee_trusted:
            self.referee_prototype = _ema_unit(
                self.referee_prototype,
                feature,
                self.config.referee_prototype_momentum,
            )

    def _goalkeeper_team(
        self,
        goalkeeper: DetectionObservation,
        observations: Mapping[int, DetectionObservation],
        assignments: Mapping[int, TeamAssignment],
    ) -> str:
        # First preference: calibrated pitch coordinates from PnLCalib. Determine
        # which outfield team is defending the nearby goal from current team medians.
        if goalkeeper.pitch_xy is not None:
            gx, gy = goalkeeper.pitch_xy
            team_xs: Dict[str, List[float]] = {TEAM_A: [], TEAM_B: []}
            for track_id, item in observations.items():
                if track_id == goalkeeper.track_id or item.pitch_xy is None:
                    continue
                assignment = assignments.get(track_id)
                if assignment is None or assignment.role != PLAYER:
                    continue
                if assignment.team in (TEAM_A, TEAM_B) and assignment.confidence >= 0.55:
                    team_xs[assignment.team].append(item.pitch_xy[0])

            if team_xs[TEAM_A] and team_xs[TEAM_B]:
                med_a = float(np.median(team_xs[TEAM_A]))
                med_b = float(np.median(team_xs[TEAM_B]))
                if gx <= self.config.goalkeeper_goal_line_distance_m:
                    return TEAM_A if med_a < med_b else TEAM_B
                if gx >= 105.0 - self.config.goalkeeper_goal_line_distance_m:
                    return TEAM_A if med_a > med_b else TEAM_B

            # If team medians are temporarily unavailable, use nearby reliable
            # outfield players on the pitch.
            nearby: List[Tuple[float, str]] = []
            for track_id, item in observations.items():
                if track_id == goalkeeper.track_id or item.pitch_xy is None:
                    continue
                assignment = assignments.get(track_id)
                if assignment is None or assignment.team not in (TEAM_A, TEAM_B):
                    continue
                dx = item.pitch_xy[0] - gx
                dy = item.pitch_xy[1] - gy
                distance = math.hypot(dx, dy)
                if distance <= self.config.goalkeeper_neighbor_radius_m:
                    nearby.append((distance, assignment.team))
            nearby.sort(key=lambda pair: pair[0])
            votes = Counter(team for _, team in nearby[:5])
            if votes:
                team, count = votes.most_common(1)[0]
                if count >= self.config.goalkeeper_min_neighbor_votes:
                    return team

        return UNKNOWN

    def _update_global_prototypes(
        self,
        valid: Sequence[Tuple[DetectionObservation, np.ndarray]],
        fused: Mapping[int, np.ndarray],
        colours: Mapping[int, np.ndarray],
        assignments: Mapping[int, TeamAssignment],
    ) -> None:
        if not self.is_ready:
            return
        for item, _ in valid:
            assignment = assignments[item.track_id]
            if (
                assignment.role == PLAYER
                and assignment.team in (TEAM_A, TEAM_B)
                and assignment.confidence >= self.config.min_prototype_update_confidence
                and not assignment.id_switch_suspected
                and self._state(item.track_id).referee_votes < self.config.referee_team_prototype_guard_votes
            ):
                self.team_prototypes[assignment.team] = _ema_unit(
                    self.team_prototypes[assignment.team],
                    fused[item.track_id],
                    self.config.prototype_momentum,
                )
                self.team_colour_prototypes[assignment.team] = _ema_unit(
                    self.team_colour_prototypes.get(assignment.team),
                    colours[item.track_id],
                    self.config.prototype_momentum,
                )

    def _best_team_similarity(self, feature: np.ndarray) -> float:
        if not self.is_ready:
            return -1.0
        return max(
            _cosine_similarity(feature, self.team_prototypes[TEAM_A]),
            _cosine_similarity(feature, self.team_prototypes[TEAM_B]),
        )


    def _best_team_colour_similarity(self, colour: np.ndarray) -> float:
        if TEAM_A not in self.team_colour_prototypes or TEAM_B not in self.team_colour_prototypes:
            return 1.0
        return max(
            _cosine_similarity(colour, self.team_colour_prototypes[TEAM_A]),
            _cosine_similarity(colour, self.team_colour_prototypes[TEAM_B]),
        )

    def _inside_pitch(self, pitch_xy: Optional[Tuple[float, float]]) -> bool:
        if pitch_xy is None:
            return False
        x, y = pitch_xy
        m = max(0.0, self.config.pitch_role_margin_m)
        return (
            -m <= x <= self.config.pitch_length_m + m
            and -m <= y <= self.config.pitch_width_m + m
        )

    def _near_goal_line(self, pitch_xy: Optional[Tuple[float, float]]) -> bool:
        if pitch_xy is None or not self._inside_pitch(pitch_xy):
            return False
        x, _ = pitch_xy
        d = self.config.goalkeeper_goal_line_distance_m
        return x <= d or x >= self.config.pitch_length_m - d

    def _expire_stale_tracks(self, frame_index: int) -> None:
        stale = [
            track_id
            for track_id, state in self.track_states.items()
            if state.last_frame >= 0 and frame_index - state.last_frame > self.config.stale_track_frames
        ]
        for track_id in stale:
            del self.track_states[track_id]

    @staticmethod
    def _bbox_jump_factor(
        previous: Tuple[float, float, float, float], current: Tuple[float, float, float, float]
    ) -> float:
        px1, py1, px2, py2 = previous
        cx1, cy1, cx2, cy2 = current
        pcenter = np.array([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dtype=np.float32)
        ccenter = np.array([(cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5], dtype=np.float32)
        pdiag = max(1.0, math.hypot(px2 - px1, py2 - py1))
        return float(np.linalg.norm(ccenter - pcenter) / pdiag)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.copy()
    return vector / norm


def _cosine_similarity(a: np.ndarray, b: Optional[np.ndarray]) -> float:
    if b is None:
        return -1.0
    an = _l2_normalize(a)
    bn = _l2_normalize(b)
    return float(np.clip(np.dot(an, bn), -1.0, 1.0))


def _cosine_distance(a: np.ndarray, b: Optional[np.ndarray]) -> float:
    if b is None:
        return 0.0
    return float(1.0 - _cosine_similarity(a, b))


def _ema_unit(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    current = _l2_normalize(current)
    if previous is None:
        return current.copy()
    mixed = (1.0 - alpha) * previous + alpha * current
    return _l2_normalize(mixed.astype(np.float32))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
