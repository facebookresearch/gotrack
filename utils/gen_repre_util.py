# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os

from bop_toolkit_lib import dataset_params
from pathlib import Path
from typing import Optional
import torch

from utils import (
    json_util,
    misc,
    pca_util,
    template_util,
    config,
    cluster_util,
    feature_util,
    repre_util,
    logging,
)

logger = logging.get_logger(__name__)


def generate_repre(
    bop_root_dir: Path,
    opts: config.GenRepreOpts,
    dataset: str,
    lid: int,
    device: str = "cuda",
    extractor: Optional[torch.nn.Module] = None,
) -> None:
    # Prepare a timer.
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Prepare the output folder.
    base_repre_dir = bop_root_dir / "object_repre"
    output_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a JSON file.
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # if "repre.pth" exists, skip the generation.
    repre_path = os.path.join(output_dir, "repre.pth")
    if os.path.exists(repre_path):
        logger.info(f"Representation already exists: {repre_path}")
        return
    # Prepare a feature extractor.
    if extractor is None:
        extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)

    timer.elapsed("Time for preparation")
    timer.start()

    # Build raw object representation.
    repre = repre_util.generate_raw_repre(
        bop_root_dir=bop_root_dir,
        opts=opts,
        dataset_name=dataset,
        object_lid=lid,
        extractor=extractor,
        output_dir=output_dir,
        device=device,
    )

    feat_vectors = repre.feat_vectors
    assert feat_vectors is not None

    timer.elapsed("Time for generating raw representation")

    # Optionally transform the feature vectors to a PCA space.
    if opts.apply_pca:
        timer.start()

        # Prepare a PCA projector.
        logger.info("Preparing PCA...")
        pca_projector = pca_util.PCAProjector(
            n_components=opts.pca_components, whiten=opts.pca_whiten
        )
        pca_projector.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_raw_projectors.append(pca_projector)

        # Transform the selected feature vectors to the PCA space.
        feat_vectors = pca_projector.transform(feat_vectors)

        timer.elapsed("Time for PCA")

    # Cluster features into visual words.
    if opts.cluster_features:
        timer.start()

        logger.info(f"Clustering features into {opts.cluster_num} visual words...")
        centroids, cluster_ids, centroid_distances = cluster_util.kmeans(
            samples=feat_vectors,
            num_centroids=opts.cluster_num,
            verbose=True,
        )

        # Store the clustering results in the object repre.
        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids

        # Get cluster sizes.
        unique_ids, unique_counts = torch.unique(cluster_ids, return_counts=True)

        timer.elapsed("Time for feature clustering")
        logger.info(
            f"{feat_vectors.shape[0]} feature vectors were clustered into {len(centroids)} clusters with {unique_counts.min()} to {unique_counts.max()} elements.",
        )

    # Generate template descriptors.
    if opts.template_desc_opts is not None:
        timer.start()
        if not isinstance(opts.template_desc_opts, config.TemplateDescOpts):
            template_desc_opts = config.TemplateDescOpts(**opts.template_desc_opts)
        repre.template_desc_opts = template_desc_opts

        # Calculate tf-idf descriptors.
        if template_desc_opts.desc_type == "tfidf":
            assert feat_vectors is not None
            assert repre.feat_cluster_centroids is not None
            assert repre.feat_to_cluster_ids is not None
            assert repre.feat_to_template_ids is not None
            assert repre.templates is not None

            (
                repre.template_descs,
                repre.feat_cluster_idfs,
            ) = template_util.calc_tfidf_descriptors(
                feat_vectors=feat_vectors,
                feat_words=repre.feat_cluster_centroids,
                feat_to_word_ids=repre.feat_to_cluster_ids,
                feat_to_template_ids=repre.feat_to_template_ids,
                num_templates=len(repre.templates),
                tfidf_knn_k=template_desc_opts.tfidf_knn_k,
                tfidf_soft_assign=template_desc_opts.tfidf_soft_assign,
                tfidf_soft_sigma_squared=template_desc_opts.tfidf_soft_sigma_squared,
            )

        else:
            raise ValueError(
                f"Unknown template descriptor type: {template_desc_opts.desc_type}"
            )

        timer.elapsed("Time for generating template descriptors")

    timer.start()

    # Create a PCA projector for visualization purposes (or reuse an existing one).
    if len(repre.feat_raw_projectors) and isinstance(
        repre.feat_raw_projectors[0], pca_util.PCAProjector
    ):
        repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]
    else:
        # Prepare a PCA projector.
        num_pca_dims_vis = 3
        pca_projector_vis = pca_util.PCAProjector(
            n_components=num_pca_dims_vis, whiten=False
        )
        pca_projector_vis.fit(
            feat_vectors, max_samples=opts.pca_max_samples_for_fitting
        )
        repre.feat_vis_projectors = [pca_projector_vis]

    repre.feat_vectors = feat_vectors

    timer.elapsed("Time for finding PCA for visualizations")
    timer.start()

    # Save the generated object representation.
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    repre_util.save_object_repre(repre, repre_dir)

    timer.elapsed("Time for saving the object representation")


def generate_repre_from_list(bop_root_dir: Path, opts: config.GenRepreOpts) -> None:
    # Get IDs of objects to process.
    object_lids = opts.object_lids
    if object_lids is None:
        bop_model_props = dataset_params.get_model_params(
            datasets_path=bop_root_dir, dataset_name=opts.dataset_name
        )
        object_lids = bop_model_props["obj_ids"]

    # Prepare a feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Process each image separately.
    for object_lid in object_lids:
        generate_repre(
            bop_root_dir, opts, opts.dataset_name, object_lid, device, extractor
        )

    # Return the root directory of the generated representations.
    return bop_root_dir / "object_repre" / opts.version / opts.dataset_name
