from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DeepClusterAssignment:
    segment_id: str
    cluster_id: int
    distance_to_own: float
    distance_to_other: float
    margin: float
    reliable: bool


@dataclass
class DeepClusterConfig:
    min_samples_per_segment: int = 3
    pca_dimensions: int = 24

    min_cluster_fraction: float = 0.25
    max_cluster_fraction: float = 0.75

    max_iterations: int = 40
    min_margin: float = 0.08


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return np.zeros_like(v)
    return v / norm


def reduce_embeddings(
    matrix: np.ndarray,
    dimensions: int,
) -> np.ndarray:
    """
    Reduce deep embeddings while preserving the natural variance structure.

    IMPORTANT:
    We intentionally do NOT whiten PCA components.

    Whitening scales every retained component to unit variance. In appearance
    embeddings that can amplify tiny/noisy dimensions until they are as
    influential as the genuinely discriminative jersey dimensions. This caused
    the v2.9 regression test to split members of the same synthetic team across
    both clusters.

    Non-whitened PCA keeps high-variance discriminative directions dominant.
    """
    matrix = np.asarray(matrix, dtype=np.float32)

    if PCA is None:
        raise RuntimeError(
            "scikit-learn PCA is required for Team Identity v2.9."
        )

    n_samples, n_features = matrix.shape

    # For very small datasets, retaining almost every PCA component simply
    # preserves noise. Keep a conservative upper bound tied to sample count.
    safe_cap = max(
        2,
        min(
            n_features,
            max(2, n_samples // 2),
        ),
    )

    n_components = min(
        max(2, int(dimensions)),
        n_samples - 1,
        safe_cap,
    )

    if n_components < 2:
        return np.vstack(
            [l2_normalize(row) for row in matrix]
        ).astype(np.float32)

    reduced = PCA(
        n_components=n_components,
        whiten=False,
        random_state=17,
    ).fit_transform(matrix)

    # Normalize each segment embedding again so Euclidean distance behaves like
    # angular distance without destroying PCA's variance weighting.
    return np.vstack(
        [l2_normalize(row) for row in reduced]
    ).astype(np.float32)


def _farthest_pair(matrix: np.ndarray) -> tuple[int, int]:
    if len(matrix) < 2:
        raise RuntimeError("Need at least two embeddings.")

    best_distance = -1.0
    best_i = 0
    best_j = 1

    for i in range(len(matrix) - 1):
        distances = np.linalg.norm(
            matrix[i + 1:] - matrix[i],
            axis=1,
        )

        if len(distances) == 0:
            continue

        j_rel = int(np.argmax(distances))
        distance = float(distances[j_rel])
        j = i + 1 + j_rel

        if distance > best_distance:
            best_distance = distance
            best_i = i
            best_j = j

    return best_i, best_j


def balanced_two_cluster(
    matrix: np.ndarray,
    config: DeepClusterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or DeepClusterConfig()
    matrix = np.asarray(matrix, dtype=np.float32)

    n = len(matrix)

    if n < 2:
        raise RuntimeError("Need at least two segment embeddings.")

    min_size = max(
        1,
        int(np.floor(n * cfg.min_cluster_fraction)),
    )
    max_size = min(
        n - 1,
        int(np.ceil(n * cfg.max_cluster_fraction)),
    )

    if min_size > max_size:
        min_size = 1
        max_size = n - 1

    a_idx, b_idx = _farthest_pair(matrix)

    centers = np.vstack(
        [
            l2_normalize(matrix[a_idx]),
            l2_normalize(matrix[b_idx]),
        ]
    ).astype(np.float32)

    labels = np.full(n, -1, dtype=np.int32)

    for _ in range(cfg.max_iterations):
        d0 = np.linalg.norm(
            matrix - centers[0],
            axis=1,
        )
        d1 = np.linalg.norm(
            matrix - centers[1],
            axis=1,
        )

        # Positive => cluster 0 is closer.
        preference = d1 - d0

        order = np.argsort(preference)[::-1]

        preferred_zero = int(np.sum(preference >= 0.0))
        zero_count = min(
            max(preferred_zero, min_size),
            max_size,
        )

        new_labels = np.ones(n, dtype=np.int32)
        new_labels[order[:zero_count]] = 0

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

        new_centers = []

        for cluster_id in (0, 1):
            rows = matrix[labels == cluster_id]

            if len(rows) == 0:
                new_centers.append(centers[cluster_id])
            else:
                new_centers.append(
                    l2_normalize(
                        np.mean(rows, axis=0)
                    )
                )

        centers = np.vstack(new_centers).astype(np.float32)

    return labels, centers


def cluster_segment_embeddings(
    embeddings: dict[str, np.ndarray],
    sample_counts: dict[str, int],
    config: DeepClusterConfig | None = None,
) -> tuple[
    dict[str, DeepClusterAssignment],
    np.ndarray,
]:
    cfg = config or DeepClusterConfig()

    eligible = [
        sid
        for sid in sorted(embeddings)
        if sample_counts.get(sid, 0) >= cfg.min_samples_per_segment
    ]

    if len(eligible) < 2:
        raise RuntimeError(
            "Not enough deep jersey segments for clustering."
        )

    raw = np.vstack(
        [l2_normalize(embeddings[sid]) for sid in eligible]
    ).astype(np.float32)

    matrix = reduce_embeddings(
        raw,
        cfg.pca_dimensions,
    )

    labels, centers = balanced_two_cluster(
        matrix,
        cfg,
    )

    result = {}

    for sid, feature, cluster_id in zip(
        eligible,
        matrix,
        labels,
    ):
        own = float(
            np.linalg.norm(
                feature - centers[cluster_id]
            )
        )

        other_id = 1 - int(cluster_id)

        other = float(
            np.linalg.norm(
                feature - centers[other_id]
            )
        )

        margin = other - own

        result[sid] = DeepClusterAssignment(
            segment_id=sid,
            cluster_id=int(cluster_id),
            distance_to_own=own,
            distance_to_other=other,
            margin=margin,
            reliable=margin >= cfg.min_margin,
        )

    return result, centers


def map_clusters_from_votes(
    assignments: dict[str, DeepClusterAssignment],
    segment_votes: dict[str, tuple[str, float, int]],
) -> tuple[dict[int, str], float]:
    scores = {
        0: {TEAM_A: 0.0, TEAM_B: 0.0},
        1: {TEAM_A: 0.0, TEAM_B: 0.0},
    }

    for sid, assignment in assignments.items():
        vote = segment_votes.get(sid)

        if vote is None:
            continue

        team, ratio, samples = vote

        if team not in {TEAM_A, TEAM_B}:
            continue

        weight = max(
            0.0,
            float(ratio) * max(1, int(samples)),
        )

        scores[assignment.cluster_id][team] += weight

    direct = scores[0][TEAM_A] + scores[1][TEAM_B]
    swapped = scores[0][TEAM_B] + scores[1][TEAM_A]

    if direct >= swapped:
        mapping = {0: TEAM_A, 1: TEAM_B}
        best, second = direct, swapped
    else:
        mapping = {0: TEAM_B, 1: TEAM_A}
        best, second = swapped, direct

    confidence = (
        best - second
    ) / max(1e-6, best + second)

    return mapping, float(confidence)
