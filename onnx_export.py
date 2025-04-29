import torch
from torch import nn
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.misc import fill_holes_in_mask_scores
import onnx
 
class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed  # [1,1,256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model._prepare_backbone_features
 
    @torch.no_grad()
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.image_encoder(
            image
        )  # {"vision_features","vision_pos_enc","backbone_fpn"}
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
 
        vision_pos_enc = backbone_out["vision_pos_enc"]  # 有3个tensor
        backbone_fpn = backbone_out["backbone_fpn"]  # 有3个tensor
        pix_feat = backbone_out["vision_features"]  # 有1个tensor
 
        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)
 
        (_, current_vision_feats, current_vision_pos_embeds, _) = self.prepare_backbone_features(
            expanded_backbone_out
        )
 
        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        current_vision_feat2 = current_vision_feat.reshape(64, 64, 1, 256).permute(
            2, 3, 0, 1
        )  # [1,256,64,64]
 
        # flatten HWxNxC -> NxCxHxW
        high_res_features_0 = (
            current_vision_feats[0].reshape(256, 256, 1, 32).permute(2, 3, 0, 1)
        )  # [1, 32, 256, 256]
        high_res_features_1 = (
            current_vision_feats[1].reshape(128, 128, 1, 64).permute(2, 3, 0, 1)
        )  # [1, 64, 128, 128]
 
        # pix_feat              [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        # current_vision_feat   [1, 256, 64, 64]
        # current_vision_pos_embed2 [4096, 1, 256]
        return (
            pix_feat,
            high_res_features_0,
            high_res_features_1,
            current_vision_feat2,
            current_vision_pos_embeds[-1],
        )
 

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc
 
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,  # [num_labels,num_points,2]
        point_labels: torch.Tensor,  # [num_labels,num_points]
        # frame_size: torch.Tensor,   # [2]
        image_embed: torch.Tensor,  # [1,256,64,64]
        high_res_feats_0: torch.Tensor,  # [1, 32, 256, 256]
        high_res_feats_1: torch.Tensor,  # [1, 64, 128, 128]
    ):
        frame_size = [256, 256]
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}
 
        batch_size = point_coords.size()[0]
        if True:
            image_embed = image_embed.repeat(batch_size, 1, 1, 1)
            high_res_feats_0 = high_res_feats_0.repeat(batch_size, 1, 1, 1)
            high_res_feats_1 = high_res_feats_1.repeat(batch_size, 1, 1, 1)
        high_res_feats = [high_res_feats_0, high_res_feats_1]
 
        sam_outputs = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=True,
        )
        (
            _,
            _,
            _,
            low_res_masks,  # [1,1,256,256]
            high_res_masks,  # [1,1,1024,1024]
            obj_ptr,  # [1,256]
            _,
        ) = sam_outputs
        # 处理高分辨率mask
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        # 填洞
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, 8)
        # 还原到原图大小
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(frame_size[0], frame_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return obj_ptr, mask_for_mem, pred_mask


class CombinedSAM2(nn.Module): 
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(sam_model)
        self.image_decoder = ImageDecoder(sam_model)
        self.point_coords = torch.randn(20, 2, 2).cpu()
        self.point_labels = torch.randn(20, 2).cpu()

    def forward(self, image):
        """
        # pix_feat              [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        # current_vision_feat   [1, 256, 64, 64]
        # current_vision_pos_embed2 [4096, 1, 256]
        return (
            pix_feat,
            high_res_features_0,
            high_res_features_1,
            current_vision_feat2,
            current_vision_pos_embeds[-1],
        )"""
        image_embed, high_res_feats_0, high_res_feats_1, _, _ = self.image_encoder(image)
        point_coords = self.point_coords
        point_labels = self.point_labels
        _, _, masks = self.image_decoder(
            point_coords=point_coords,
            point_labels=point_labels,
            image_embed=image_embed,
            high_res_feats_0=high_res_feats_0,
            high_res_feats_1=high_res_feats_1,
        )
        return masks
        

 
 
def export_image_encoder(model, onnx_path):
    input_img = torch.randn(1, 3, 1024, 1024).cpu()
    out = model(input_img)
    output_names = [
        "pix_feat",
        "high_res_feat0",
        "high_res_feat1",
        "vision_feats",
        "vision_pos_embed",
    ]
    torch.onnx.export(
        model,
        input_img,
        onnx_path + "image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    # # 简化模型, tmd将我的输出数量都简化掉一个，sb
    # original_model = onnx.load(onnx_path+"image_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path + "image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_encoder.onnx model is valid!")
 

def export_image_decoder(model, onnx_path):
    point_coords = torch.randn(20, 2, 2).cpu()
    point_labels = torch.randn(20, 2).cpu()
    # point_coords = torch.randn(1,2,2).cpu()
    # point_labels = torch.randn(1,2).cpu()
    # frame_size = torch.tensor([1024,1024],dtype=torch.int64)
    image_embed = torch.randn(1, 256, 64, 64).cpu()
    high_res_feats_0 = torch.randn(1, 32, 256, 256).cpu()
    high_res_feats_1 = torch.randn(1, 64, 128, 128).cpu()
    # pix_feat              [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
    out = model(
        point_coords=point_coords,
        point_labels=point_labels,
        #    frame_size = frame_size,
        image_embed=image_embed,
        high_res_feats_0=high_res_feats_0,
        high_res_feats_1=high_res_feats_1,
    )
    # input_name = ["point_coords","point_labels","frame_size","image_embed","high_res_feats_0","high_res_feats_1"]
    input_name = [
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
    ]
    output_name = ["obj_ptr", "mask_for_mem", "pred_mask"]
    dynamic_axes = {
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        # "obj_ptr": {0: "batch_size"},
        # "mask_for_mem": {0: "batch_size"},
        # "pred_mask": {0: "batch_size"}
    }
    torch.onnx.export(
        model,
        #    (point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1),
        (point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1),
        onnx_path + "mask_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_name,
        output_names=output_name,
        dynamic_axes=dynamic_axes,
    )
    # 简化模型,
    # original_model = onnx.load(onnx_path+"image_decoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_decoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"mask_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("mask_decoder.onnx model is valid!")
 

def export_combined(mode, onnx_path):
    input_img = torch.randn(1, 3, 1024, 1024).cpu()
    out = mode(input_img)
    output_names = ["masks"]
    torch.onnx.export(
        mode,
        input_img,
        onnx_path + "combined_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path + "combined_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("combined_model.onnx model is valid!")
    
# ****************************************************************************
 
if __name__ == "__main__":
    # model_dir = sys.argv[1] + '/'
    # model_type = "base_plus"
    # sam2_url = f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_{model_type}.pt"
 
    # model_config_file =  "sam2_hiera_{}.yaml".format(model_type)
    # model_checkpoints_file = f"{model_dir}/sam2_hiera_{model_type}.pt"
    # if os.path.exists(model_checkpoints_file):
    #     print(f"SAM2 checkpoint found in {model_checkpoints_file}")
    # else:
    #     os.system(f"wget -O {model_checkpoints_file} {sam2_url}")
    model_dir = "./"
    model_type = "tiny"
    model_config_file = "sam2.1_hiera_{}.yaml".format(model_type[0])
    model_checkpoints_file = "checkpoints/sam2.1_hiera_{}.pt".format(model_type)
    sam2_model = build_sam2(model_config_file, model_checkpoints_file, device="cpu")
 
    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder, model_dir)
 
    image_decoder = ImageDecoder(sam2_model).cpu()
    export_image_decoder(image_decoder, model_dir)
    
    combined_model = CombinedSAM2(sam2_model).cpu()
    export_combined(combined_model, model_dir)
    
# def main():
#     model_type = ["tiny", "small", "large", "base+"][3]
#     onnx_output_path = "checkpoints/{}/".format(model_type)
#     model_config_file = "sam2_hiera_{}.yaml".format(model_type)
#     model_checkpoints_file = "checkpoints/sam2_hiera_{}.pt".format(model_type)
 
#     parser = argparse.ArgumentParser(description="Export SAM2")
#     parser.add_argument("--outdir", type=str, default=onnx_output_path, required=False, help="path")
#     parser.add_argument(
#         "--config", type=str, default=model_config_file, required=False, help="*.yaml"
#     )
#     parser.add_argument(
#         "--checkpoint", type=str, default=model_checkpoints_file, required=False, help="*.pt"
#     )
#     args = parser.parse_args()
#     sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")
 
#     image_encoder = ImageEncoder(sam2_model).cpu()
#     export_image_encoder(image_encoder,args.outdir)
 
#     image_decoder = ImageDecoder(sam2_model).cpu()
#     export_image_decoder(image_decoder, args.outdir)
 
#     # mem_attention = MemAttention(sam2_model).cpu()
#     # export_memory_attention(mem_attention,args.outdir)
 
#     # mem_encoder   = MemEncoder(sam2_model).cpu()
#     # export_memory_encoder(mem_encoder,args.outdir)