import os
import math
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.cm as cm

class ONNXSamInfer:
    def __init__(self, model_path: str):
        """
        Initialize the ONNXRuntime session.
        Args:
            model_path: Path to the exported ONNX model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, image_path: str, output_path: str):
        """
        Run inference on the given image and save a grid of overlay images,
        one per mask, laid out in a square grid.
        Args:
            image_path: Path to the input image.
            output_path: Path to save the output grid image.
        """
        # 1) Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 1024), resample=Image.BILINEAR)
        img_np = np.array(image).astype(np.float32) / 255.0  # [H,W,3]
        img_input = img_np.transpose(2, 0, 1)[None, ...]     # [1,3,1024,1024]

        # 2) ONNX inference
        ort_inputs = {self.input_name: img_input}
        masks_out = self.session.run([self.output_name], ort_inputs)[0]
        # masks_out: [1, N, 1024, 1024]
        masks = masks_out[0]  # [N, H, W]

        # 3) Threshold logits to binary masks
        bin_masks = masks > 0.2
        num_masks = bin_masks.shape[0]
        num_masks = 10
        print(f"Number of masks: {masks.shape}")

        # 4) Create overlay images for each mask
        overlay_imgs = []
        colormap = cm.get_cmap('jet', num_masks)
        for i in range(num_masks):
            overlay = img_np.copy()
            mask = bin_masks[i]
            color = np.array(colormap(i)[:3])
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
            overlay_img = (overlay * 255).astype(np.uint8)
            overlay_imgs.append(Image.fromarray(overlay_img))

        # 5) Compute grid size (square-ish)
        grid_cols = int(math.ceil(math.sqrt(num_masks)))
        grid_rows = int(math.ceil(num_masks / grid_cols))
        cell_w, cell_h = 1024, 1024
        grid_w = grid_cols * cell_w
        grid_h = grid_rows * cell_h
        grid_img = Image.new('RGB', (grid_w, grid_h))

        # 6) Paste each overlay into the grid
        for idx, img in enumerate(overlay_imgs[:]):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_img.paste(img, (col * cell_w, row * cell_h))

        # 7) Save the final grid image
        grid_img.save(output_path)

# Example usage:
infer = ONNXSamInfer('sam2_onnx.onnx')
infer.infer('original_image.png', 'output_grid.png')