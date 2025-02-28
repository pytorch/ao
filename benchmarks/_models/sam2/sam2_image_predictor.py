# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from benchmarks._models.sam2.modeling.sam2_base import SAM2Base
from benchmarks._models.sam2.utils.misc import get_image_size
from benchmarks._models.sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor(torch.nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self._image_dtype = torch.float32
        self._transforms_device = "cpu"

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image, torch.Tensor],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        self._orig_hw = [get_image_size(image)]

        if isinstance(image, torch.Tensor):
            # from torchvision.transforms.v2 import functional as F
            # input_image = F.to_dtype(image, torch.float32, scale=True)
            input_image = image
        else:
            input_image = self._transforms.to_tensor(image)

        # NOTE: Doing these transforms on the GPU changes the numerics
        input_image = input_image.to(device=self._transforms_device)
        input_image = self._transforms.transforms(input_image)
        input_image = input_image.to(device=self.device)
        # input_image = self._transforms.transforms(input_image)
        input_image = input_image[None, ...].to(dtype=self._image_dtype)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        with torch.autograd.profiler.record_function("forward_image"):
            backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = list(map(get_image_size, image_list))
        with torch.autograd.profiler.record_function("forward_batch"):
            # Transform the image to the form expected by the model
            # img_batch = self._transforms.forward_batch(image_list)
            image_list = [
                self._transforms.to_tensor(img) if isinstance(img, np.ndarray) else img
                for img in image_list
            ]
            image_list = [self._transforms.transforms(img) for img in image_list]
            img_batch = torch.stack(image_list, dim=0)
            img_batch = img_batch.to(self.device)
            img_batch = img_batch.to(self._image_dtype)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        with torch.autograd.profiler.record_function("forward_image"):
            backbone_out = self.model.forward_image(img_batch)
        with torch.autograd.profiler.record_function("_prepare_backbone_features"):
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        return_type: str = "numpy",
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        if return_type not in ["numpy", "torch"]:
            raise ValueError(
                f"Expected return_type to be either numpy or torch, but got {return_type}"
            )
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            if return_type == "numpy":
                masks = masks.squeeze(0).float().detach().cpu().numpy()
                iou_predictions = (
                    iou_predictions.squeeze(0).float().detach().cpu().numpy()
                )
                low_res_masks = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            # NOTE: Need these additional clones to prevent overwriting tensor output of CUDAGraph
            all_masks.append(masks.clone())
            all_ious.append(iou_predictions.clone())
            all_low_res_masks.append(low_res_masks.clone())

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        return_type: str = "numpy",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if return_type not in ["numpy", "torch"]:
            raise ValueError(
                f"Expected return_type to be either numpy or torch, but got {return_type}"
            )
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        if return_type == "torch":
            return (
                masks.squeeze(0),
                iou_predictions.squeeze(0),
                low_res_masks.squeeze(0),
            )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = (
                torch.as_tensor(point_coords, dtype=torch.float)
                .pin_memory()
                .to(self.device, non_blocking=True)
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = (
                torch.as_tensor(point_labels, dtype=torch.int)
                .pin_memory()
                .to(self.device, non_blocking=True)
            )
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        with torch.autograd.profiler.record_function("_predict_masks"):
            high_res_feats = self._features["high_res_feats"]
            image_embed = self._features["image_embed"]
            image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            high_res_feats_input = [
                feat_level[img_idx].unsqueeze(0).clone()
                # for feat_level in self._features["high_res_feats"]
                for feat_level in high_res_feats
            ]
            image_embed_input = image_embed[img_idx].unsqueeze(0).clone()
            assert boxes is None
            assert mask_input is None
            assert multimask_output is True
            low_res_masks, iou_predictions = self._predict_masks(
                [t.contiguous() for t in high_res_feats_input],
                image_embed_input.contiguous(),
                image_pe.contiguous(),
                point_coords.contiguous(),
                point_labels.contiguous(),
                boxes=boxes,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )
            # img_idx=img_idx)
        with torch.autograd.profiler.record_function("_predict_masks_postprocess"):
            masks, low_res_masks = self._predict_masks_postprocess(
                low_res_masks, img_idx, return_logits
            )
            return masks, iou_predictions, low_res_masks

    def _predict_masks(
        self,
        high_res_feats_input,
        image_embed,
        image_pe,
        point_coords,
        point_labels,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ):
        # NOTE: img_idx causes unnecessary recompilations, because
        # the int guard will fail otherwise.
        #   img_idx: int = -1):
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        with torch.autograd.profiler.record_function("self.model.sam_prompt_encoder"):
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=mask_input,
            )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        # high_res_features = [
        #     feat_level[img_idx].unsqueeze(0).clone()
        #     # for feat_level in self._features["high_res_feats"]
        #     for feat_level in high_res_feats_input
        # ]
        high_res_features = high_res_feats_input
        with torch.autograd.profiler.record_function("self.model.sam_mask_decoder"):
            # if not multimask_output:
            #     raise ValueError("Expected multimask_output.")
            # if batched_mode:
            #     raise ValueError("Did not expected repeat_image.")
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                # image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0).clone(),
                # image_embeddings=image_embed[img_idx].unsqueeze(0).clone(),
                image_embeddings=image_embed,
                # image_pe=self.model.sam_prompt_encoder.get_dense_pe().clone(),
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings.clone(),
                dense_prompt_embeddings=dense_embeddings.clone(),
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

        return low_res_masks, iou_predictions

    def _predict_masks_postprocess(
        self, low_res_masks, img_idx, return_logits, channel_1=False
    ):
        # TODO: Might want to defer this until after data["iou_preds"] > self.pred_iou_thresh
        with torch.autograd.profiler.record_function(
            "self._transforms.postprocess_masks"
        ):
            # Upscale the masks to the original image resolution
            if channel_1:
                masks = self._transforms.postprocess_masks_1_channel(
                    low_res_masks, self._orig_hw[img_idx], self._image_dtype
                )
            else:
                masks = self._transforms.postprocess_masks(
                    low_res_masks, self._orig_hw[img_idx], self._image_dtype
                )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
