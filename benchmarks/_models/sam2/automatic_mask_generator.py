# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from torchao._models.sam2.modeling.sam2_base import SAM2Base
from torchao._models.sam2.sam2_image_predictor import SAM2ImagePredictor
from torchao._models.sam2.utils.amg import (
    MaskData,
    _mask_to_rle_pytorch_2_0,
    _mask_to_rle_pytorch_2_1,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge_torch,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from torchao._models.sam2.utils.misc import (
    crop_image,
    get_image_size,
)


class SAM2AutomaticMaskGenerator(torch.nn.Module):
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        **kwargs,
    ) -> None:
        """
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.
        """
        super().__init__()
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output

        # Store a reference to these on the model so I can overwrite them
        # with compile annotation if desired

        self.calculate_stability_score = calculate_stability_score
        self.batched_mask_to_box = batched_mask_to_box

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2AutomaticMaskGenerator): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        return self._encode_masks(mask_data)

    def _encode_masks(self, mask_data):
        mask_data["rles"] = _mask_to_rle_pytorch_2_1(mask_data["rles"])
        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    @torch.no_grad()
    def generate_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        data = self._generate_masks_batch(images)
        return [self._encode_masks(d) for d in data]

    def _generate_masks(self, image: Union[np.ndarray, torch.Tensor]) -> MaskData:
        orig_size = get_image_size(image)
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = None
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            if data is None:
                data = crop_data
            else:
                data.cat(crop_data)

        return self._deduplicate_masks(crop_boxes, data)

    def _deduplicate_masks(self, crop_boxes, data):
        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def _generate_masks_batch(self, images: List[np.ndarray]) -> List[MaskData]:
        all_orig_size = []
        all_crop_boxes = []
        all_layer_idxs = []
        for image in images:
            orig_size = get_image_size(image)
            all_orig_size.append(orig_size)
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )
            all_crop_boxes.append(crop_boxes)
            all_layer_idxs.append(layer_idxs)

        all_data = self._process_crop_batch(
            images, all_crop_boxes, all_layer_idxs, all_orig_size
        )

        return [
            self._deduplicate_masks(crop_boxes, data)
            for (crop_boxes, data) in zip(all_crop_boxes, all_data)
        ]

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        cropped_im = crop_image(image, crop_box)
        cropped_im_size = get_image_size(cropped_im)
        with torch.autograd.profiler.record_function("set_image"):
            self.predictor.set_image(cropped_im)

        return self._process_crop_points(
            cropped_im_size, crop_layer_idx, crop_box, orig_size
        )

    def _process_crop_points(
        self, cropped_im_size, crop_layer_idx, crop_box, orig_size
    ):
        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        # data = MaskData()
        data = None
        points_per_batch = self.points_per_batch
        if self.points_per_batch is None:
            points_per_batch = len(points_for_image)
        for (points,) in batch_iterator(points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            with torch.autograd.profiler.record_function("data.cat"):
                if data is None:
                    data = batch_data
                else:
                    data.cat(batch_data)
                    del batch_data
        self.predictor.reset_predictor()
        return self._process_crop_points_dedup(data, crop_box)

    def _process_crop_points_dedup(self, data, crop_box):
        with torch.autograd.profiler.record_function("batched_nms"):
            # Remove duplicates within this crop.
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )

        with torch.autograd.profiler.record_function("filter"):
            data.filter(keep_by_nms)

        with torch.autograd.profiler.record_function("uncrop_boxes_xyxy"):
            # Return to the original image frame
            data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        with torch.autograd.profiler.record_function("uncrop_points"):
            data["points"] = uncrop_points(data["points"], crop_box)
        with torch.autograd.profiler.record_function("crop_boxes"):
            data["crop_boxes"] = torch.tensor(
                [crop_box for _ in range(len(data["rles"]))]
            )

        return data

    def _process_crop_batch(
        self,
        images: List[np.ndarray],
        all_crop_boxes: List[List[int]],
        all_layer_idxs: List[int],
        all_orig_size_compact: List[Tuple[int, ...]],
    ) -> List[MaskData]:
        all_image = []
        all_crop_box = []
        all_layer_idx = []
        all_orig_size = []
        for image, orig_size, crop_boxes, layer_idxs in zip(
            images, all_orig_size_compact, all_crop_boxes, all_layer_idxs
        ):
            # Iterate over image crops
            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                all_image.append(image)
                all_crop_box.append(crop_box)
                all_layer_idx.append(layer_idx)
                all_orig_size.append(orig_size)

        # # TODO: NOTE: Calling process_crop in a loop like this might be an issue, because the predictor is stateful

        all_cropped_im = [
            crop_image(image, crop_box)
            for image, crop_box in zip(all_image, all_crop_box)
        ]

        with torch.autograd.profiler.record_function("set_batch_image"):
            self.predictor.set_image_batch(all_cropped_im)

        i = 0
        batch_features = self.predictor._features
        all_crop_data = []
        for cropped_im, crop_box, layer_idx, orig_size in zip(
            all_cropped_im, all_crop_box, all_layer_idx, all_orig_size
        ):
            cropped_im_size = get_image_size(cropped_im)
            self.predictor._orig_hw = [get_image_size(cropped_im)]
            self.predictor._features = {
                "image_embed": batch_features["image_embed"][i].unsqueeze(0),
                "high_res_feats": [
                    b[i].unsqueeze(0) for b in batch_features["high_res_feats"]
                ],
            }
            i += 1
            self.predictor._is_image_set = True

            # TODO: Batch mask_to_rle_pytorch_2 calls
            # TODO: Specialize for rle-only return (specify which keys you want in data)

            # all_crop_data.append(self._process_crop_points(cropped_im_size, layer_idx, crop_box, orig_size))

            crop_layer_idx = layer_idx

            # Get points for this crop
            points_scale = np.array(cropped_im_size)[None, ::-1]
            points_for_image = self.point_grids[crop_layer_idx] * points_scale

            # Generate masks for this crop in batches
            points_per_batch = self.points_per_batch
            if self.points_per_batch is None:
                points_per_batch = len(points_for_image)

            all_batch_iterator_data = []
            with torch.autograd.profiler.record_function("all _process_batch"):
                for (points,) in batch_iterator(points_per_batch, points_for_image):
                    # batch_data = self._process_batch(
                    #     points, cropped_im_size, crop_box, orig_size, normalize=True
                    # )

                    im_size = cropped_im_size
                    normalize = True

                    orig_h, orig_w = orig_size

                    orig_box = [0, 0, orig_w, orig_h]
                    orig_box_torch = torch.as_tensor(
                        orig_box, dtype=torch.float, device=self.predictor.device
                    )
                    crop_box_torch = torch.as_tensor(
                        crop_box, dtype=torch.float, device=self.predictor.device
                    )
                    data = self._process_batch_fullgraph(
                        points,
                        im_size,
                        crop_box,
                        crop_box_torch,
                        orig_size,
                        normalize,
                        orig_box_torch,
                    )
                    all_batch_iterator_data.append(data)
                self.predictor.reset_predictor()

            result_data = None
            with torch.autograd.profiler.record_function("all mask_to_rle_pytorch_2"):
                for data in all_batch_iterator_data:
                    # Compress to RLE
                    data["masks"] = uncrop_masks(
                        data["masks"], crop_box, orig_h, orig_w
                    )
                    # TODO: Capture all these masks in a single NT for mask_to_rle_pytorch_2
                    # or at a minimum create a mask_to_rle_pytorch_2_list and use loops
                    # to cause a single DtoH sync
                    data["rles"] = _mask_to_rle_pytorch_2_0(data["masks"])
                    del data["masks"]

                    batch_data = data
                    with torch.autograd.profiler.record_function("data.cat"):
                        if result_data is None:
                            result_data = batch_data
                        else:
                            result_data.cat(batch_data)
                            del batch_data
                self.predictor.reset_predictor()

            all_crop_data.append(self._process_crop_points_dedup(result_data, crop_box))

        i = 0
        all_data = []
        for _, _, crop_boxes, layer_idxs in zip(
            images, all_orig_size, all_crop_boxes, all_layer_idxs
        ):
            data = None
            for _, _ in zip(crop_boxes, layer_idxs):
                if data is None:
                    data = all_crop_data[i]
                else:
                    data.cat(all_crop_data[i])
                i += 1
            all_data.append(data)
        return all_data

    def _process_batch_fullgraph(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        crop_box_torch: torch.Tensor,
        orig_size: Tuple[int, ...],
        normalize: bool,
        orig_box_torch: torch.Tensor,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(points, dtype=torch.float32).pin_memory()
        points = points.to(device=self.predictor.device, non_blocking=True)
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        with torch.autograd.profiler.record_function("_predict"):
            # NOTE: Just leaving this for reference. predict was split to
            # allow earlier filtering by predicted iou
            # masks, iou_preds, low_res_masks = self.predictor._predict(
            masks = None
            high_res_feats = self.predictor._features["high_res_feats"]
            image_embed = self.predictor._features["image_embed"]
            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe().clone()
            assert (
                self.multimask_output
            ), "Currently require multimask_output set to True"
            high_res_feats_input = [
                feat_level[-1].unsqueeze(0).clone()
                # for feat_level in self._features["high_res_feats"]
                for feat_level in high_res_feats
            ]
            image_embed_input = image_embed[-1].unsqueeze(0).clone()
            low_res_masks, iou_preds = self.predictor._predict_masks(
                [t.contiguous() for t in high_res_feats_input],
                image_embed_input.contiguous(),
                image_pe.contiguous(),
                in_points[:, None, :].contiguous(),
                in_labels[:, None].contiguous(),
                boxes=None,
                mask_input=None,
                multimask_output=self.multimask_output,
                # img_idx=-1,
            )

        x0, y0, _, _ = crop_box
        points = points.repeat_interleave(3 if masks is None else masks.shape[1], dim=0)

        if not self.use_m2m:
            with torch.autograd.profiler.record_function("thresh and filter"):
                # Filter by predicted IoU
                if self.pred_iou_thresh > 0.0:
                    keep_mask = iou_preds.flatten(0, 1) > self.pred_iou_thresh
                    keep_index = keep_mask.nonzero(as_tuple=True)[0]
                    low_res_masks = low_res_masks.flatten(0, 1).unsqueeze(1)
                    low_res_masks = low_res_masks[keep_index]
                    iou_preds = iou_preds.flatten(0, 1).unsqueeze(1)[keep_index]
                    points = points[keep_index]
        if masks is None:
            masks, low_res_mask = self.predictor._predict_masks_postprocess(
                low_res_masks, -1, True, channel_1=low_res_masks.size(1) == 1
            )

        # Serialize predictions and store in MaskData
        with torch.autograd.profiler.record_function("MaskData"):
            data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=points,
                low_res_masks=low_res_masks.flatten(0, 1),
            )
        del masks

        keep_mask = None

        if not self.use_m2m:
            # NOTE: This is left just for reference. We filter earlier for this case to save
            # on compute within the expensive _predict_masks_postprocess
            # with torch.autograd.profiler.record_function("thresh and filter"):
            #     # Filter by predicted IoU
            #     if self.pred_iou_thresh > 0.0:
            #         keep_mask = data["iou_preds"] > self.pred_iou_thresh
            #         # TODO: Might need this for correctness due to calculate_stability_score IoU?
            #         data.filter(keep_mask)

            with torch.autograd.profiler.record_function("calculate_stability_score"):
                # Calculate and filter by stability score
                data["stability_score"] = self.calculate_stability_score(
                    data["masks"], self.mask_threshold, self.stability_score_offset
                )
            with torch.autograd.profiler.record_function("stability_score_thresh"):
                if self.stability_score_thresh > 0.0:
                    keep_mask = data["stability_score"] >= self.stability_score_thresh
                    keep_index = keep_mask.nonzero(as_tuple=True)[0]
                    data.filter(keep_index)
        else:
            # One step refinement using previous mask predictions
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = self.calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        with torch.autograd.profiler.record_function(
            "Threshold masks and calculate boxes"
        ):
            # Threshold masks and calculate boxes
            data["masks"] = data["masks"] > self.mask_threshold
            data["boxes"] = self.batched_mask_to_box(data["masks"])

        with torch.autograd.profiler.record_function("is_box_near_crop_edge"):
            # Filter boxes that touch crop boundaries
            keep_mask = ~is_box_near_crop_edge_torch(
                data["boxes"],
                crop_box,
                crop_box_torch,
                orig_box_torch,
            )

        with torch.autograd.profiler.record_function("filter(keep_mask)"):
            keep_index = keep_mask.nonzero(as_tuple=True)[0]
            data.filter(keep_index)

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        orig_box = [0, 0, orig_w, orig_h]
        orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float)
        orig_box_torch = orig_box_torch.pin_memory()
        orig_box_torch = orig_box_torch.to(
            device=self.predictor.device, non_blocking=True
        )

        crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float)
        crop_box_torch = crop_box_torch.pin_memory()
        crop_box_torch = crop_box_torch.to(
            device=self.predictor.device, non_blocking=True
        )

        data = self._process_batch_fullgraph(
            points,
            im_size,
            crop_box,
            crop_box_torch,
            orig_size,
            normalize,
            orig_box_torch,
        )

        with torch.autograd.profiler.record_function("uncrop_masks"):
            # Compress to RLE
            data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
            data["rles"] = _mask_to_rle_pytorch_2_0(data["masks"])
            del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        # TODO: This doesn't use the possibly compiled self.batched_mask_to_box
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        new_masks = []
        new_iou_preds = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            best_masks, best_iou_preds, _ = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
        masks = torch.cat(new_masks, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0)
