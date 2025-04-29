import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2

class SAM2ONNXWrapper(nn.Module):
    def __init__(self, sam_model: SAM2Base, 
                 point_coords: torch.Tensor,  # shape [N,2]
                 point_labels: torch.Tensor,  # shape [N]
                 multimask_output: bool = False):
        """
        Wraps SAM2Base to accept only images and output final masks.
        
        Arguments:
          sam_model: your loaded SAM2Base instance (with .sam_prompt_encoder and .sam_mask_decoder)
          point_coords: tensor of shape [N,2], absolute pixel coordinates in [0,1023]
          point_labels: tensor of shape [N], labels in {0,1}
          multimask_output: whether to output multiple masks per point
        """
        super().__init__()
        self.model = sam_model
        self.multimask_output = multimask_output

        # register prompts as constant buffers (will be embedded inside ONNX graph)
        # and normalize them by image size
        norm_coords = point_coords.to(torch.float32) / sam_model.image_size
        self.register_buffer("point_coords", norm_coords[None, ...])   # [1, N, 2]
        self.register_buffer("point_labels", point_labels[None, ...])  # [1, N]
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        B = image.shape[0]

        # 1) Backbone + FPN convs
        backbone_out = self.model.image_encoder(image)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        # 2) Prepare and flatten features
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

        # 3) Reconstruct the [B,C,H,W] embeddings
        #   a) final low-res
        feat_top = vision_feats[-1]       # [H*W, B, C]
        H, W    = feat_sizes[-1]
        image_embeddings = (
            feat_top.permute(1, 2, 0)
            .contiguous()
            .view(B, -1, H, W)
        )

        #   b) high_res for SAM decoder
        high_res_feats = []
        for lvl_feat, (h, w) in zip(vision_feats[:-1], feat_sizes[:-1]):
            high_res_feats.append(
                lvl_feat.permute(1, 2, 0)
                        .contiguous()
                        .view(B, -1, h, w)
            )

        # 4) Prompt embedding (points only)
        pts = self.point_coords.expand(B, -1, -1)   # [B, N, 2]
        lbl = self.point_labels.expand(B, -1)       # [B, N]
        sparse_emb, dense_emb = self.model.sam_prompt_encoder(
            points=(pts, lbl), boxes=None, masks=None
        )

        # 5) Mask decoder
        image_pe = self.model.sam_prompt_encoder.get_dense_pe().expand(B, -1, -1, -1)
        masks_logits, iou_preds, *_ = self.model.sam_mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            repeat_image=True,
            high_res_features=high_res_feats,
        )

        # 6) Pick single mask & upsample to 1024×1024
        masks, _ = self.model.sam_mask_decoder._dynamic_multimask_via_stability(
            masks_logits, iou_preds
        )
        masks = torch.clamp(masks, -32.0, 32.0)
        masks = F.interpolate(masks, size=(1024, 1024),
                            mode="bilinear", align_corners=False)

        return masks  # [B,1,1024,1024]

import numpy as np
def get_points(num_points):
    img_w, img_h = 1024, 1024
    point_w = img_w // num_points
    point_h = img_h // num_points

    x_coords = np.arange(0, img_w, point_w)
    y_coords = np.arange(0, img_h, point_h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.stack((xx.flatten(), yy.flatten()), axis=-1)  # [N, 2]
    labels = np.arange(points.shape[0]+1)[1:]
    return torch.from_numpy(points), torch.from_numpy(labels)


# ======================
# Example of exporting:
# ======================
if __name__ == "__main__":
    # assume you have:
    sam_model      = build_sam2(
        config_file="sam2.1_hiera_l.yaml", 
        ckpt_path="checkpoints/sam2.1_hiera_large.pt",
        device='cpu')  # your loaded SAM2Base
    pts, lbls      =        get_points(10)     # your chosen fixed N points
    wrapper = SAM2ONNXWrapper(sam_model, pts, lbls, multimask_output=False).eval()

    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        "sam2_onnx.onnx",
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["masks"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "masks":  {0: "batch_size"},
        },
    )
    print("Exported sam2_onnx.onnx successfully.")
