# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Tuple

import faiss
import torch
from utils import torch_helpers


def kmeans(
    samples: torch.Tensor,
    num_centroids: int,
    num_iter: int = 50,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K-means clustering.

    Ref:
    [1] https://github.com/facebookresearch/faiss/wiki/FAQ#how-z
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

    Args:
        samples: Samples of shape (num_samples, num_dims).
        num_centroids: Number of centroids/clusters.
        num_iter: Number of k-means iterations.
        verbose: Whether to print progress.
    Returns:
        A tuple of (centroids, cluster_ids, centroid_distances).
    """

    # Whether to use GPU.
    gpu = True if samples.device.type == "cuda" else False

    # Create a k-means object.
    num_dims = samples.shape[1]
    kmeans = faiss.Kmeans(
        num_dims,
        num_centroids,
        niter=num_iter,
        gpu=gpu,
        verbose=True,
        seed=0,
        spherical=False,
    )

    # faiss.Kmeans requires the samples to be on CPU.
    samples_cpu = samples.cpu().numpy()

    # Cluter the samples.
    kmeans.train(samples_cpu)

    # Get per-sample cluster assignments.
    centroid_distances, cluster_ids = kmeans.index.search(samples_cpu, 1)

    # Reshape from (num_samples, 1) to (num_samples).
    centroid_distances = centroid_distances.squeeze(axis=-1)
    cluster_ids = cluster_ids.squeeze(axis=-1)

    # Move the result to the original device.
    device = samples.device
    return (
        torch_helpers.array_to_tensor(kmeans.centroids).to(device),
        torch_helpers.array_to_tensor(cluster_ids).to(torch.int32).to(device),
        torch_helpers.array_to_tensor(centroid_distances).to(device),
    )
