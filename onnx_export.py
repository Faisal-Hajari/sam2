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
        self.register_buffer("point_coords", norm_coords)       # shape [N,2]
        self.register_buffer("point_labels", point_labels)      # shape [N]
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """
        image: [B,3,1024,1024]
        returns: [B, N, 1024, 1024]
        """
        B = image.shape[0]
        N = self.point_coords.shape[0]

        # 1) Encode once:
        backbone_out = self.model.image_encoder(image)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

        # 2) Reconstruct [B,C,H,W] and [B,C_i,h_i,w_i]:
        H, W = feat_sizes[-1]
        feat_top        = vision_feats[-1]  # [H*W, B, C]
        image_emb       = feat_top.permute(1,2,0).contiguous().view(B, -1, H, W)
        high_res_feats  = []
        for lvl, (h,w) in zip(vision_feats[:-1], feat_sizes[:-1]):
            high_res_feats.append(
                lvl.permute(1,2,0).contiguous().view(B, -1, h, w)
            )

        # 3) Tile image embeddings for each point:
        #    from [B,C,H,W] -> [B*N, C, H, W]
        image_emb_tiled = image_emb.unsqueeze(1).expand(B, N, -1, -1, -1)
        image_emb_tiled = image_emb_tiled.reshape(B*N, image_emb.shape[1], H, W)

        #    similarly tile each high_res_feats[level]
        high_res_tiled = []
        for feat in high_res_feats:
            b,c,h,w = feat.shape
            tmp = feat.unsqueeze(1).expand(b, N, c, h, w).reshape(B*N, c, h, w)
            high_res_tiled.append(tmp)

        # 4) Prepare per-point prompts as a big batch:
        #    coords: [N,2] -> [B,N,2] -> [B*N,2]
        N = self.point_coords.shape[0]
        # coords: [N,2] -> [1,N,2] -> [B,N,2] -> [B*N,2]
        pts = self.point_coords.unsqueeze(0).expand(B, N, 2).reshape(B*N, 2)

        # labels: [N] -> [1,N] -> [B,N] -> [B*N]
        lbl = self.point_labels.unsqueeze(0).expand(B, N).reshape(B*N)

        # 5) Embed that batch of single-point prompts:
        sparse_emb, dense_emb = self.model.sam_prompt_encoder(
            points=(pts[:,None,:], lbl[:,None]),  # each batch elt has exactly one point
            boxes=None, masks=None
        )
        # dense_emb: [B*N, embed_dim, h, w]

        # 6) Decode all B*N masks at once:
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        
        print("image_pe.shape", image_pe.shape)
        print("sparse_emb.shape", sparse_emb.shape)
        print("dense_emb.shape", dense_emb.shape)
        
        masks_logits, iou_preds, *_ = self.model.sam_mask_decoder.predict_masks(
            image_embeddings=image_emb_tiled,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            repeat_image=False,
            high_res_features=high_res_tiled,
        )
        print("masks_logits.shape", masks_logits.shape)
        print("iou_preds.shape", iou_preds.shape)
        # 7) Collapse to single mask (dynamic multimask) per prompt:
        masks, _ = self.model.sam_mask_decoder._dynamic_multimask_via_stability(
            masks_logits, iou_preds
        )
        # masks: [B*N, 1, h, w]

        # 8) Upsample to 1024×1024:
        masks = torch.clamp(masks, -32.0, 32.0)
        masks = F.interpolate(masks, size=(1024,1024),
                            mode="bilinear", align_corners=False)
        # now [B*N, 1,1024,1024]

        # 9) reshape back to [B, N, 1024,1024]
        masks = masks.reshape(B, N, 1024, 1024)

        return masks

import numpy as np
def get_points(num_points):
    img_w, img_h = 1024, 1024
    point_w = img_w // num_points
    point_h = img_h // num_points

    x_coords = np.arange(0, img_w, point_w)
    y_coords = np.arange(0, img_h, point_h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.stack((xx.flatten(), yy.flatten()), axis=-1)  # [N, 2]
    labels = np.ones(points.shape[0])
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
    
    print(pts)
    print(lbls)
    exit()
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

# this seems to work and is generating mutliple masks. but are thoes masks independent ? like no point influnce the other ? 