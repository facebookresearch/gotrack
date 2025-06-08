# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

# This script is used to visualize the dataloader.
# pip install imageio[ffmpeg] imageio[pyav]
# python -m scripts.vis_dataset mode=pose_refinement dataset_name=lmo coarse_pose_method=foundpose
from pathlib import Path
import distinctipy
import hydra
import imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from utils import data_util, net_util, render_vis_util, renderer_builder  # noqa: F401
from utils.logging import get_logger
import warnings
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def visualize(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if cfg.mode == "pose_refinement":
        # Define the dataloader using dataset_name, coarse_pose_method
        assert cfg.dataset_name is not None
        assert cfg.coarse_pose_method is not None
        cfg.data.dataloader.dataset_name = cfg.dataset_name
        cfg.data.dataloader.coarse_pose_method = cfg.coarse_pose_method
        dataset = instantiate(cfg.data.dataloader)
        save_dir = (
            Path(cfg.save_dir)
            / f"vis_{cfg.mode}"
            / f"{cfg.dataset_name}_{cfg.coarse_pose_method}"
        )
    else:
        assert cfg.object_name is not None
        assert cfg.sequence_name is not None
        cfg.data.rbot.dataloader.object_name = cfg.object_name
        cfg.data.rbot.dataloader.sequence_name = cfg.sequence_name
        dataset = instantiate(cfg.data.rbot.dataloader)
        save_dir = (
            Path(cfg.save_dir)
            / f"vis_{cfg.mode}"
            / f"{cfg.object_name}_{cfg.sequence_name}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialze the renderer.
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(
        renderer_type=renderer_type, model_path=dataset.dp_model["model_tpath"]
    )

    # Initialize the colors for the objects.
    logger.info(f"Number of objects: {len(dataset.models)}")
    max_obj_ids = max(dataset.models.keys())
    colors = distinctipy.get_colors(max_obj_ids + 1)

    count = 0
    vis_ims = []
    for scene_observation in tqdm(dataset):
        scene_id = scene_observation.scene_id
        im_id = scene_observation.im_id
        objects_anno = scene_observation.objects_anno
        object_lids = [obj.lid for obj in objects_anno]
        object_poses_m2w = [obj.pose for obj in objects_anno]

        vis_im = render_vis_util.vis_posed_meshes_of_objects(
            base_image=scene_observation.image,
            camera_c2w=scene_observation.camera,
            object_poses_m2w=object_poses_m2w,
            object_lids=object_lids,
            renderer=renderer,
            object_colors=[colors[lid] for lid in object_lids],
        )[0]

        im_name = f"{scene_id:06d}_{im_id:06d}.png"
        vis_im_path = save_dir / im_name
        vis_ims.append(vis_im)
        vis_im = Image.fromarray(np.uint8(vis_im))
        vis_im.save(vis_im_path)
        logger.info(f"Saved {vis_im_path}")

        if count == 10:
            break
        count += 1
    video_path = save_dir / "video.mp4"
    imageio.mimwrite(video_path, vis_ims, fps=5, macro_block_size=8)
    logger.info(f"Saved video to {video_path}")


if __name__ == "__main__":
    visualize()
