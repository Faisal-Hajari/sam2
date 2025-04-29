import os
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SAM2OnnxInference:
    def __init__(self, model_path):
        """
        Initialize the SAM2 ONNX inference class.
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Expected image size for the model
        self.img_size = 1024
        
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for model inference.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            np.ndarray: Preprocessed image
            np.ndarray: Original image for visualization
        """
        # Read image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 1024x1024
        img = cv2.resize(original_img, (self.img_size, self.img_size))
        
        # Normalize to [0, 1] and convert to RGB
        img = img.astype(np.float32) / 255.0
        
        # Convert to NCHW format (batch, channels, height, width)
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return img, original_img
    
    def run_inference(self, input_img):
        """
        Run inference on the preprocessed image.
        
        Args:
            input_img (np.ndarray): Preprocessed image in NCHW format
            
        Returns:
            np.ndarray: Mask predictions
        """
        # Run inference
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: input_img}
        )
        
        # Get the mask predictions (shape: [1, 1, 1024, 1024])
        masks = outputs[0]
        print(f"Mask shape: {masks.shape}")
        return masks
    
    def postprocess_masks(self, masks):
        """
        Postprocess the mask predictions.
        
        Args:
            masks (np.ndarray): Mask predictions from the model
            
        Returns:
            np.ndarray: Binary masks
        """
        # Apply sigmoid to convert logits to probabilities
        masks = 1 / (1 + np.exp(-masks))
        
        # Threshold to get binary masks (adjust threshold as needed)
        binary_masks = (masks > 0.5).astype(np.uint8)
        
        return binary_masks
    
    def visualize_and_save(self, original_img, binary_masks, output_path):
        """
        Visualize and save the segmentation results.
        
        Args:
            original_img (np.ndarray): Original image
            binary_masks (np.ndarray): Binary masks
            output_path (str): Path to save the output image
        """
        # Resize original image if needed
        if original_img.shape[:2] != (self.img_size, self.img_size):
            original_img = cv2.resize(original_img, (self.img_size, self.img_size))
        
        # Create RGB mask overlay
        mask_overlay = np.zeros_like(original_img)
        
        # Extract the first mask (shape: [1024, 1024])
        mask = binary_masks[0, 0]
        
        # Create a colored overlay for the mask (using red color with some transparency)
        mask_overlay[mask > 0] = [255, 0, 0]  # Red color
        
        # Blend original image with mask overlay
        alpha = 0.5  # Transparency factor
        blended = cv2.addWeighted(original_img, 1, mask_overlay, alpha, 0)
        
        # Save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(blended)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Also save the raw mask for potential further use
        mask_path = os.path.splitext(output_path)[0] + "_mask.png"
        cv2.imwrite(mask_path, mask * 255)
        
        print(f"Visualization saved to: {output_path}")
        print(f"Raw mask saved to: {mask_path}")
    
    def process_image(self, image_path, output_path=None):
        """
        Process an image and save the visualization with segmentation masks.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the output image. If None, 
                                         will be derived from input path.
        """
        # Set default output path if not specified
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(os.path.dirname(image_path), f"{name}_segmented{ext}")
        
        # Preprocess image
        input_img, original_img = self.preprocess_image(image_path)
        
        # Run inference
        masks = self.run_inference(input_img)
        
        # Postprocess masks
        binary_masks = self.postprocess_masks(masks)
        
        # Visualize and save results
        self.visualize_and_save(original_img, binary_masks, output_path)
        
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize the inference class
    model_path = "sam2_onnx.onnx"  # Path to your exported ONNX model
    inference = SAM2OnnxInference(model_path)
    
    # Process an image
    image_path = "original_image.png"  # Path to your input image
    output_path = inference.process_image(image_path)
    
    print(f"Processing complete. Output saved to: {output_path}")