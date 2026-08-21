import numpy as np

from core.team_deep_embedding_v29 import (
    TEAM_A,
    TEAM_B,
    DeepClusterConfig,
    balanced_two_cluster,
    cluster_segment_embeddings,
    map_clusters_from_votes,
)


def test_balanced_clustering_prevents_38_3_collapse():
    rng = np.random.default_rng(7)

    group_a = rng.normal(
        loc=0.0,
        scale=0.12,
        size=(20, 8),
    )
    group_a[:, 0] += 2.0

    group_b = rng.normal(
        loc=0.0,
        scale=0.12,
        size=(21, 8),
    )
    group_b[:, 1] += 2.0

    matrix = np.vstack([group_a, group_b]).astype(np.float32)

    labels, _ = balanced_two_cluster(
        matrix,
        DeepClusterConfig(
            min_cluster_fraction=0.25,
            max_cluster_fraction=0.75,
        ),
    )

    counts = np.bincount(labels, minlength=2)

    assert counts.min() >= 10
    assert counts.max() <= 31


def test_deep_segment_clustering_separates_groups():
    rng = np.random.default_rng(8)

    embeddings = {}
    counts = {}

    for i in range(8):
        v = rng.normal(0, 0.03, 16)
        v[0] += 1.0
        embeddings[f"a{i}"] = v.astype(np.float32)
        counts[f"a{i}"] = 5

    for i in range(8):
        v = rng.normal(0, 0.03, 16)
        v[1] += 1.0
        embeddings[f"b{i}"] = v.astype(np.float32)
        counts[f"b{i}"] = 5

    assignments, _ = cluster_segment_embeddings(
        embeddings,
        counts,
    )

    a_clusters = {
        assignments[f"a{i}"].cluster_id
        for i in range(8)
    }
    b_clusters = {
        assignments[f"b{i}"].cluster_id
        for i in range(8)
    }

    assert len(a_clusters) == 1
    assert len(b_clusters) == 1
    assert a_clusters != b_clusters


def test_mapping_only_names_clusters_after_clustering():
    embeddings = {
        "a": np.array([1, 0, 0, 0], dtype=np.float32),
        "b": np.array([0.9, 0.1, 0, 0], dtype=np.float32),
        "c": np.array([0, 1, 0, 0], dtype=np.float32),
        "d": np.array([0.1, 0.9, 0, 0], dtype=np.float32),
    }

    counts = {k: 5 for k in embeddings}

    assignments, _ = cluster_segment_embeddings(
        embeddings,
        counts,
        DeepClusterConfig(pca_dimensions=2),
    )

    votes = {
        "a": (TEAM_A, 0.9, 10),
        "b": (TEAM_A, 0.8, 8),
        "c": (TEAM_B, 0.9, 10),
        "d": (TEAM_B, 0.8, 8),
    }

    mapping, confidence = map_clusters_from_votes(
        assignments,
        votes,
    )

    assert mapping[assignments["a"].cluster_id] == TEAM_A
    assert mapping[assignments["c"].cluster_id] == TEAM_B
    assert confidence > 0
