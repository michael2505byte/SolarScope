"""
Models Module for Combined Building and Solar Panel Detection
Includes:
- BuildingDetectionModel: DeepLab V3+ for building detection
- SolarPanelDetectionModel: Wrapper for RCNN-based solar panel detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from PIL import Image
import cv2
import os


class DeepLabV3PlusForBuildings(nn.Module):
    """
    DeepLab v3+ model adapted for binary building segmentation
    """
    def __init__(self, num_classes=1, pretrained=False):
        super(DeepLabV3PlusForBuildings, self).__init__()
        
        # Load DeepLab v3+ with ResNet-50 backbone
        self.model = deeplabv3_resnet50(pretrained=pretrained)
        
        # Modify classifier for binary segmentation
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Modify auxiliary classifier as well
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Add sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Get the segmentation output
        result = self.model(x)['out']
        
        # Apply sigmoid for binary classification
        result = self.sigmoid(result)
        
        return result


class BuildingDetectionModel:
    """
    Wrapper class for building detection model with preprocessing and postprocessing
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the building detection model
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            device: Device to run inference on (cuda/mps/cpu). Auto-detects if None.
        """
        # Set device with Apple Silicon (MPS) support
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Building Model - Using device: {self.device}")
        
        # Initialize model architecture
        self.model = DeepLabV3PlusForBuildings(num_classes=1, pretrained=False)
        
        # Load trained weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print(f"Building model loaded successfully from {model_path}")
            except RuntimeError as e:
                print(f"Warning: Loading with strict=False due to: {e}")
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Building model loaded successfully (with strict=False)")
        except Exception as e:
            print(f"Error loading building model: {e}")
            raise
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Reduce memory usage
        torch.set_grad_enabled(False)
        
        # Image preprocessing parameters
        self.image_size = 256  # Reduced from 512 to save memory
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to the input image
            
        Returns:
            preprocessed_tensor: Tensor ready for model input
            original_image: Original PIL image
            original_size: Original image size (width, height)
        """
        # Load image
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size  # (width, height)
        
        # Convert to tensor
        image_tensor = transforms.ToTensor()(original_image)
        
        # Resize to model input size
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize
        image_tensor = self.normalize(image_tensor)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_image, original_size
    
    def postprocess_prediction(self, prediction, original_size, threshold=0.5):
        """
        Postprocess model prediction
        
        Args:
            prediction: Model output tensor
            original_size: Original image size (width, height)
            threshold: Threshold for binary classification
            
        Returns:
            mask: Binary mask as PIL Image
            confidence_map: Confidence map as numpy array
        """
        # Remove batch dimension and move to CPU
        prediction = prediction.squeeze().cpu().detach().numpy()
        
        # Resize to original size
        prediction_resized = cv2.resize(
            prediction,
            original_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create binary mask
        binary_mask = (prediction_resized > threshold).astype(np.uint8) * 255
        
        # Convert to PIL Image
        mask_image = Image.fromarray(binary_mask, mode='L')
        
        return mask_image, prediction_resized
    
    def create_overlay(self, original_image, mask, alpha=0.5):
        """
        Create an overlay of the prediction on the original image
        
        Args:
            original_image: Original PIL Image
            mask: Binary mask PIL Image
            alpha: Transparency factor for overlay
            
        Returns:
            overlay_image: PIL Image with prediction overlay
        """
        # Convert images to numpy arrays
        original_np = np.array(original_image)
        mask_np = np.array(mask)
        
        # Create colored mask (red for buildings)
        colored_mask = np.zeros_like(original_np)
        colored_mask[:, :, 0] = mask_np  # Red channel for buildings
        
        # Blend original image with colored mask
        overlay_np = cv2.addWeighted(original_np, 1, colored_mask, alpha, 0)
        
        # Convert back to PIL Image
        overlay_image = Image.fromarray(overlay_np)
        
        return overlay_image
    
    def calculate_statistics(self, prediction, threshold=0.5):
        """
        Calculate statistics from the prediction
        
        Args:
            prediction: Prediction numpy array (0-1 range)
            threshold: Threshold for binary classification
            
        Returns:
            dict: Dictionary with statistics
        """
        binary_pred = prediction > threshold
        
        total_pixels = prediction.size
        building_pixels = np.sum(binary_pred)
        building_percentage = (building_pixels / total_pixels) * 100
        confidence_mean = np.mean(prediction[binary_pred]) if building_pixels > 0 else 0.0
        
        return {
            'total_pixels': total_pixels,
            'building_pixels': building_pixels,
            'building_percentage': building_percentage,
            'confidence_mean': confidence_mean
        }
    
    @torch.no_grad()
    def predict(self, image_path, threshold=0.5):
        """
        Perform building detection on an image
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binary classification (default: 0.5)
            
        Returns:
            dict: Dictionary containing prediction results and statistics
        """
        try:
            # Preprocess image
            input_tensor, original_image, original_size = self.preprocess_image(image_path)
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            prediction = self.model(input_tensor)
            
            # Postprocess prediction
            mask_image, confidence_map = self.postprocess_prediction(
                prediction, original_size, threshold
            )
            
            # Create overlay
            overlay_image = self.create_overlay(original_image, mask_image)
            
            # Calculate statistics
            stats = self.calculate_statistics(confidence_map, threshold)
            
            # Prepare results
            results = {
                'prediction_image': mask_image,
                'overlay_image': overlay_image,
                'confidence_map': confidence_map,
                'building_pixels': stats['building_pixels'],
                'total_pixels': stats['total_pixels'],
                'building_percentage': stats['building_percentage'],
                'confidence_mean': stats['confidence_mean'],
                'image_size': original_size
            }
            
            return results
            
        finally:
            # Clean up memory after prediction
            if 'input_tensor' in locals():
                del input_tensor
            if 'prediction' in locals():
                del prediction
            # Clear cache for CUDA or MPS
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass  # MPS cache clearing not critical
            import gc
            gc.collect()


class SolarPanelDetectionModel:
    """
    Wrapper class for solar panel detection using RCNN with geoai
    Note: This class mainly serves as a placeholder/wrapper since
    geoai.object_detection is called directly in the Flask app
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the solar panel detection model
        
        Args:
            model_path: Path to the trained RCNN model
            device: Device to run inference on (cuda/mps/cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.model_path = model_path
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Solar panel model not found at {model_path}")
        
        print(f"Solar Panel Model - Path verified: {model_path}")
        print(f"Solar Panel Model - Using device: {self.device}")
    
    def get_model_path(self):
        """Return the model path for use with geoai"""
        return self.model_path


# Test function
if __name__ == "__main__":
    print("Models Module - Building and Solar Panel Detection")
    print("="*60)
    print("This module contains:")
    print("  1. BuildingDetectionModel - DeepLab V3+ for buildings")
    print("  2. SolarPanelDetectionModel - RCNN wrapper for solar panels")
    print("="*60)
