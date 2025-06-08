# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import numpy as np
import torch
from ultralytics import yolo  # noqa
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from utils import logging, structs
from model import config
import torch.nn.functional as F

logger = logging.get_logger(__name__)


class FastSAM(YOLO):
    """FastSAM model for segmentation and detection."""

    def __init__(
        self,
        opts: config.FastSAMOpts,
    ):
        logger.info(f"FastSAM: Loading model from {opts.model_path}")
        YOLO.__init__(
            self,
            opts.model_path,
        )
        self.opts = opts
        overrides_params = {
            "iou": opts.iou_threshold,
            "conf": opts.conf_threshold,
            "max_det": opts.max_det,
            "imgsz": opts.im_width_size,
            "verbose": opts.verbose,
            "mode": "predict",
            "save": False,
        }
        self.segmentor = yolo.v8.segment.SegmentationPredictor(
            overrides=overrides_params, _callbacks=self.callbacks
        )

    def set_device(self, device: torch.device, verbose=False):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.segmentor.args.half &= device.type != "cpu"
        self.segmentor.model = AutoBackend(
            self.model,
            device=device,
            dnn=self.segmentor.args.dnn,
            data=self.segmentor.args.data,
            fp16=self.segmentor.args.half,
            fuse=True,
            verbose=verbose,
        )
        self.segmentor.device = device
        self.segmentor.model.eval()
        logger.info(f"FastSAM: setup model at device {device} done!")

    def __call__(self, source=None, stream=False):
        return self.segmentor(source=source, stream=stream)

    def postprocess_resize(self, detections, orig_size, update_boxes=False):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes:
            scale = orig_size[1] / self.opts.im_width_size
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"][:, [0, 2]] = torch.clamp(
                detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
            )
            detections["boxes"][:, [1, 3]] = torch.clamp(
                detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
            )
        return detections

    @torch.no_grad()
    def generate_detection_proposals(self, image: np.ndarray) -> structs.Collection:
        if self.opts.im_width_size is not None:
            orig_size = image.shape[:2]
        fastSAM_outputs = self(image)

        # Since the detection has been made in the input opts.im_width_size, we convert back to the original size.
        masks = fastSAM_outputs[0].masks.data
        boxes = fastSAM_outputs[0].boxes.data[:, :4]  # two lasts:  confidence and class

        # define class data
        detections = {
            "masks": masks,
            "boxes": boxes,
        }
        if self.opts.im_width_size is not None:
            detections = self.postprocess_resize(detections, orig_size)

        outputs = structs.Collection()
        outputs.masks = detections["masks"]
        outputs.boxes = detections["boxes"]
        return outputs
