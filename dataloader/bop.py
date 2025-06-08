# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


import os
from typing import List, Optional
from bop_toolkit_lib import inout, pycoco_utils
from bop_toolkit_lib import dataset_params
from utils import data_util, misc, structs
from utils.logging import get_logger
import numpy as np
from tqdm import tqdm
from dataloader.base import GoTrackDataset

logger = get_logger(__name__)


class BopDataset(GoTrackDataset):
    """Dataloader for BOP datasets."""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        is_localization_task: bool = True,
        use_default_detections: bool = False,
        coarse_pose_method: Optional[str] = None,
    ) -> None:
        super().__init__(root_dir=root_dir)
        self.dataset_name = dataset_name
        self.is_localization_task = is_localization_task
        self.coarse_pose_method = coarse_pose_method
        self.use_default_detections = use_default_detections
        logger.info(
            f"BOP dataset: is_localization_task={is_localization_task} and use_default_detections={use_default_detections}."
        )

        # Initialize dataset parameters.
        self.models = None
        self.models_info = None
        self.models_vertices = None
        self.scene_cameras = None
        self.dp_model = dataset_params.get_model_params(
            self.root_dir, self.dataset_name
        )
        self.dp_split = None

        if self.coarse_pose_method is not None:
            # Load dataset from coarse pose estimates.
            self.coarse_poses_per_image = None
            self.coarse_poses_file_name = None

            # Get the dataset info, models and models info.
            self.load_dataset_info_from_coarse_pose()
            logger.info(f"Loaded dataset info from {coarse_pose_method}'s estimates.")
        else:
            self.targets = None
            self.targets_per_image = None
            self.load_dataset_info_from_target_file()
            logger.info("Loaded dataset info from target file.")
            if self.use_default_detections:
                self.load_default_detections()
                logger.info("Loaded dataset info from target file.")

        # Load the models.
        self.models = {}
        self.models_vertices = {}
        logger.info("Loading object models...")
        for obj_id in self.dp_model["obj_ids"]:
            self.models[obj_id] = inout.load_ply(
                self.dp_model["model_tpath"].format(obj_id=obj_id)
            )
            # Sample vertices.
            max_vertices = 1000  # followed FoundPose.
            self.models_vertices[obj_id] = np.random.permutation(
                self.models[obj_id]["pts"]
            )[:max_vertices]
            self.models_vertices[obj_id] = self.models_vertices[obj_id].astype(
                np.float32
            )
        # Load models info.
        self.models_info = inout.load_json(
            self.dp_model["models_info_path"], keys_to_int=True
        )

    def load_dataset_info_from_target_file(self) -> None:
        # List all the files in the target directory.
        dataset_dir = self.root_dir / self.dataset_name

        # List all test folders.
        test_folders = [
            folder for folder in list(dataset_dir.glob("test*")) if folder.is_dir()
        ]
        if len(test_folders) == 1:
            folder_name = test_folders[0].name
            split_type = folder_name.split("_")[-1] if "_" in folder_name else None
        else:
            split_type = None
        logger.info(f"Dataset: {self.dataset_name}, split_type:{split_type}")

        # Load dataset parameters.
        self.dp_split = dataset_params.get_split_params(
            self.root_dir, self.dataset_name, split="test", split_type=split_type
        )

        # If multiple files are found, use the bop_version to select the correct one.
        target_files = list(dataset_dir.glob("test_targets_bop*.json"))
        if len(target_files) > 1:
            bop_version = "bop19"
            if self.dataset_name in ["hot3d", "hopev2", "handal"]:
                bop_version = "bop24"
            target_files = [f for f in target_files if bop_version in str(f)]
        assert len(target_files) == 1
        logger.info(f"Loading target file: {target_files[0]}")
        targets = inout.load_json(str(target_files[0]))
        self.targets_per_image = {}
        for item in targets:
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            if (scene_id, im_id) not in self.targets_per_image:
                self.targets_per_image[(scene_id, im_id)] = {}

            # For localization tasks, the target objects need to be specified.
            if self.is_localization_task:
                assert "obj_id" in item, (
                    f"obj_id not found in {item}. "
                    f"For localization tasks, the target objects need to be specified but not found in target file. "
                    f"Please check localization task is available for {self.dataset_name}."
                )
            if "obj_id" in item:
                obj_id = int(item["obj_id"])
                inst_count = int(item["inst_count"])
                self.targets_per_image[(scene_id, im_id)][obj_id] = inst_count

    def load_default_detections(self) -> None:
        # Get the dataset group.
        if self.dataset_name in [
            "lmo",
            "tless",
            "tudl",
            "icbin",
            "itodd",
            "hb",
            "ycbv",
        ]:
            detection_folder_name = "classic_bop23_model_based_unseen/cnos-fastsam"
        elif self.dataset_name in ["hot3d", "hopev2", "handal"]:
            detection_folder_name = "h3_bop24_model_based_unseen/cnos-fastsam-with-mask"
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for default detections."
            )

        # Find the default detections.
        default_detections_dir = (
            self.root_dir / "default_detections" / detection_folder_name
        )

        default_detection_files = list(default_detections_dir.glob("*.json"))
        default_detection_file = [
            f for f in default_detection_files if self.dataset_name in str(f)
        ]
        assert len(default_detection_file) == 1, (
            f"Expected 1 file, found {len(default_detection_file)}."
        )
        default_detection_file = default_detection_file[0]
        logger.info(f"Loading default detections: {default_detection_file}")

        #  Load default detections.
        self.default_detections = inout.load_json(str(default_detection_file))

        # Organize both target and coarse poses per (scene_id, im_id)
        # Loading the default detections of whole dataset is not memory efficient (HOT3D has 200k detections),
        # so we store only the index of the detections for each image.
        self.default_detections_per_image = {}
        for idx, item in tqdm(
            enumerate(self.default_detections),
            desc="Loading default detections",
            unit="item",
        ):
            scene_id, im_id = int(item["scene_id"]), int(item["image_id"])
            run_time = item["time"]
            if (scene_id, im_id) not in self.default_detections_per_image:
                self.default_detections_per_image[(scene_id, im_id)] = {
                    "time": run_time,
                    "idx_detections": [],
                }
            else:
                assert (
                    run_time
                    == self.default_detections_per_image[(scene_id, im_id)]["time"]
                )
            self.default_detections_per_image[(scene_id, im_id)][
                "idx_detections"
            ].append(idx)

    def load_dataset_info_from_coarse_pose(self) -> None:
        # Find the prediction of the coarse pose method.
        coarse_pose_dir = self.root_dir / "coarse_poses" / self.coarse_pose_method
        # List all the files in the coarse pose directory.
        coarse_pose_files = list(
            coarse_pose_dir.glob(
                f"{self.coarse_pose_method}_{self.dataset_name}-test_*.csv"
            )
        )
        assert len(coarse_pose_files) == 1, (
            f"Expected 1 file for {self.coarse_pose_method}/{self.dataset_name}, found {len(coarse_pose_files)}."
        )
        coarse_pose_path = coarse_pose_files[0]

        """Load dataset info from the coarse pose file."""
        # Parse info about the coarse pose method and the dataset from the filename.
        result_name = os.path.splitext(os.path.basename(coarse_pose_path))[0]
        result_info = result_name.split("_")
        dataset_info = result_info[1].split("-")
        dataset = str(dataset_info[0])
        split = str(dataset_info[1])
        split_type = str(dataset_info[2]) if len(dataset_info) > 2 else None
        logger.info(f"Dataset: {dataset}, split:{split}")

        # Save the result_name for later use.
        self.coarse_poses_file_name = result_name

        # Load dataset parameters.
        self.dp_split = dataset_params.get_split_params(
            self.root_dir, dataset, split, split_type
        )

        # Load coarse poses
        coarse_poses = inout.load_bop_results(coarse_pose_path)

        # Organize both target and coarse poses per (scene_id, im_id)
        self.coarse_poses_per_image = {}
        for item in tqdm(coarse_poses, desc="Loading coarse poses", unit="item"):
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            obj_id = int(item["obj_id"])
            run_time = item["time"]
            R, t = np.asarray(item["R"]), np.asarray(item["t"])
            est = structs.ObjectAnnotation(
                dataset=self.dataset_name,
                lid=int(obj_id),
                pose=structs.ObjectPose(R=R, t=t),
            )

            if (scene_id, im_id) not in self.coarse_poses_per_image:
                self.coarse_poses_per_image[(scene_id, im_id)] = {
                    "time": run_time,
                    "objects": [],
                }
            else:
                assert (
                    run_time == self.coarse_poses_per_image[(scene_id, im_id)]["time"]
                )
            self.coarse_poses_per_image[(scene_id, im_id)]["objects"].append(est)

    def load_scene_camera(self, scene_id: int) -> None:
        assert self.dp_split is not None
        """Load scene camera (intrinsic) for the given scene_id."""
        tpath_keys = dataset_params.scene_tpaths_keys(
            self.dp_split["eval_modality"], self.dp_split["eval_sensor"], scene_id
        )
        scene_camera = inout.load_scene_camera(
            self.dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
        )
        return data_util.parse_chunk_cameras(scene_camera)

    def load_detections_from_indexes(
        self, detection_indexes: int
    ) -> List[structs.ObjectAnnotation]:
        """Load detections for the given indexes."""
        ests = []
        for idx in detection_indexes:
            item = self.default_detections[idx]
            obj_id = int(item["category_id"])
            bbox = item["bbox"]
            mask = pycoco_utils.rle_to_binary_mask(item["segmentation"])
            # Convert bbox to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            bbox = [x1, y1, x1 + w, y1 + h]
            est = structs.ObjectAnnotation(
                dataset=self.dataset_name,
                lid=int(obj_id),
                boxes_modal=bbox,
                masks_modal=mask,
                score=float(item["score"]),
            )
            ests.append(est)
        return ests

    def __len__(self) -> int:
        """Return the number of test images in the dataset."""
        if self.coarse_pose_method is None:
            return len(self.targets_per_image)
        else:
            return len(self.coarse_poses_per_image)

    def __getitem__(self, idx: int) -> structs.SceneObservation:
        """Return the test image and its corresponding target."""
        # Get the scene_id and im_id for the current index.
        if self.coarse_pose_method is None:
            scene_id, im_id = list(self.targets_per_image.keys())[idx]
        else:
            scene_id, im_id = list(self.coarse_poses_per_image.keys())[idx]
        # Load the image.
        if "gray" in self.dp_split["im_modalities"]:
            # ITODD includes grayscale instead of RGB images.
            image_path = self.dp_split["gray_tpath"].format(
                scene_id=scene_id, im_id=im_id
            )
        elif "aria" in self.dp_split["im_modalities"]:
            # HOT3D includes Aria, Quest3 images which are captured by different sensors.
            eval_modality = self.dp_split["eval_modality"](scene_id)
            eval_sensor = self.dp_split["eval_sensor"](scene_id)
            image_path = self.dp_split[f"{eval_modality}_{eval_sensor}_tpath"].format(
                scene_id=scene_id, im_id=im_id
            )
        else:
            image_path = self.dp_split["rgb_tpath"].format(
                scene_id=scene_id, im_id=im_id
            )
        # TODO: bop_toolkit defines wrong rgb extension for HANDAL.
        if self.dataset_name == "handal":
            image_path = image_path.replace(".png", ".jpg")
        image = inout.load_im(image_path)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        # Load the scene camera (intrinsics).
        scene_camera = self.load_scene_camera(scene_id)
        camera = scene_camera[im_id]
        camera.im_size = image.shape[:2]
        camera.height = image.shape[0]
        camera.width = image.shape[1]

        # By default, the dataloader is for detection tasks, only camera intrinsic is available.
        scene_observation = structs.SceneObservation(
            scene_id=scene_id,
            im_id=im_id,
            image=np.asarray(image, dtype=np.uint8),
            camera=camera,
        )
        if self.coarse_pose_method is not None:
            # Get the coarse poses for the current image.
            coarse_poses = self.coarse_poses_per_image[(scene_id, im_id)]["objects"]
            # Get run-time of coarse pose stage.
            run_time = self.coarse_poses_per_image[(scene_id, im_id)]["time"]
            objects_anno = []
            for est_id, est in enumerate(coarse_poses):
                # 6D object pose.
                pose_m2w = None
                if est.pose is not None:
                    pose_m2c = est.pose
                    trans_c2w = camera.T_world_from_eye
                    trans_m2w = np.matmul(trans_c2w, misc.get_rigid_matrix(pose_m2c))
                    pose_m2w = structs.ObjectPose(
                        R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:].reshape(3, 1)
                    )

                objects_anno.append(
                    structs.ObjectAnnotation(
                        dataset=self.dataset_name,
                        lid=est.lid,
                        pose=pose_m2w,
                    )
                )
            scene_observation = scene_observation._replace(objects_anno=objects_anno)
            scene_observation = scene_observation._replace(time=run_time)

        # Dataloader for pose estimation stage using default detections.
        if self.use_default_detections:
            # Get the default detections for the current image.
            default_detections = self.load_detections_from_indexes(
                self.default_detections_per_image[(scene_id, im_id)]["idx_detections"]
            )
            # Get run-time of coarse pose stage.
            run_time = self.default_detections_per_image[(scene_id, im_id)]["time"]
            scene_observation = scene_observation._replace(time=run_time)
            scene_observation = scene_observation._replace(
                objects_anno=default_detections,
            )

        # In case of localization tasks, the target objects are available.
        target_objects = None
        if self.coarse_pose_method is None:
            if self.is_localization_task:
                target_objects = self.targets_per_image[(scene_id, im_id)]
        return {
            "scene_observation": scene_observation,
            "target_objects": target_objects,
        }
