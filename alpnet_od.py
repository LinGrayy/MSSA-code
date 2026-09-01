"""
ProtoSAM-based Optic Disc Segmentation
Simplified interface for image-to-image prototype matching
"""

import os
import torch
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import torch.nn.functional as F

from models.ProtoSAM import ProtoSAM
from models.ProtoMedSAM import ProtoMedSAM
from models.grid_proto_fewshot import FewShotSeg
from models.SamWrapper import SamWrapper
from segment_anything.utils.transforms import ResizeLongestSide
import cv2

# ============================================================================
# Configuration
# ============================================================================

class ProtoSAMConfig:
    """Configuration for ProtoSAM model"""
    
    def __init__(self, **kwargs):
        # Model type
        self.base_model = kwargs.get('base_model', 'alpnet')
        self.protosam_sam_ver = kwargs.get('protosam_sam_ver', 'sam_h')
        self.model_name = kwargs.get('model_name', 'dinov2_l14')
        
        # Model paths
        self.sam_checkpoint = kwargs.get('sam_checkpoint', 
            '/home/psxll9/MedCLIP-SAM/checkpoint/sam_vit_h_4b8939.pth')
        self.reload_model_path = kwargs.get('reload_model_path', None)
        
        # Input
        self.input_size = kwargs.get('input_size', (672, 672))
        self.image_size = kwargs.get('image_size', (1024, 1024))
        
        # ProtoSAM settings
        self.use_bbox = kwargs.get('use_bbox', True)
        self.use_points = kwargs.get('use_points', True)
        self.use_mask = kwargs.get('use_mask', False)
        self.do_cca = kwargs.get('do_cca', True)
        self.point_mode = kwargs.get('point_mode', 'both')
        self.coarse_pred_only = kwargs.get('coarse_pred_only', True)
        self.use_neg_points = kwargs.get('use_neg_points', False)
        self.num_points_for_sam = kwargs.get('num_points_for_sam', 1)
        self.use_sam_trans = kwargs.get('use_sam_trans', True)
        self.debug = kwargs.get('debug', False)
        
        # Support set
        self.n_support = kwargs.get('n_support', 1)
        
        # Few-shot model settings (for ALPNet)
        self.proto_grid_size = kwargs.get('proto_grid_size', 8)
        self.feature_hw = kwargs.get('feature_hw', [84, 84])
        self.lora = kwargs.get('lora', 0)
        self.use_pos_enc = kwargs.get('use_pos_enc', False)
        
        # Dataset
        self.modality = kwargs.get('modality', 'opticdisc')
        self.dataset = kwargs.get('dataset', 'opticdisc')
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Seed
        self.seed = kwargs.get('seed', 42)
        
        # Model architecture
        self.model_config = {
            'align': True,
            'dinov2_loss': False,
            'use_coco_init': True,
            'which_model': self.model_name,
            'cls_name': 'grid_proto',
            'proto_grid_size': self.proto_grid_size,
            'feature_hw': self.feature_hw,
            'reload_model_path': self.reload_model_path,
            'lora': self.lora,
            'use_slice_adapter': False,
            'adapter_layers': 3,
            'debug': self.debug,
            'use_pos_enc': self.use_pos_enc,
        }


# ============================================================================
# ALPNet (Few-Shot Segmentation) Model
# ============================================================================
class ALPNetModel:
    """ALPNet few-shot segmentation model wrapper"""
    
    def __init__(self, config: ProtoSAMConfig):
        self.config = config
        self.model = None
        self._init_model()
    
    def _init_model(self):
        from models.grid_proto_fewshot import FewShotSeg
        
        print("  Loading ALPNet model...")
        self.model = FewShotSeg(
            self.config.input_size[0],
            self.config.reload_model_path,
            self.config.model_config
        )
        self.model.to(self.config.device)
        self.model.eval()
        print("  ALPNet model loaded")
    
    def predict(self,
               query_image: torch.Tensor,
               support_images: torch.Tensor,
               support_mask: torch.Tensor,
               isval: bool = True,
               val_wsize: int = 2) -> torch.Tensor:
        """
        Run ALPNet prediction.
        
        Returns:
            Predicted segmentation (H, W), values in [0, 1]
        """
        # Ensure batch dimension
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)      # (1, 3, H, W)
        if support_images.dim() == 3:
            support_images = support_images.unsqueeze(0) # (1, 3, H, W)
        if support_mask.dim() == 2:
            support_mask = support_mask.unsqueeze(0)     # (1, H, W)
        
        # Background mask = inverse of foreground
        fore_mask = support_mask.clone()
        back_mask = 1.0 - support_mask.clone()
        
        # Format for FewShotSeg
        supp_imgs = [[support_images]]    # way x shot x [B x 3 x H x W]
        fore_mask = [[fore_mask]]          # way x shot x [B x H x W]
        back_mask = [[back_mask]]          # way x shot x [B x H x W]
        qry_imgs = [query_image]           # N x [B x 3 x H x W]
        
        with torch.no_grad():
            output, _, _, _, _, _, _ = self.model(
                supp_imgs=supp_imgs,
                fore_mask=fore_mask,
                back_mask=back_mask,
                qry_imgs=qry_imgs,
                isval=isval,
                val_wsize=val_wsize,
            )
        
        # DEBUG: print output stats
        if self.config.debug:
            print(f"  ALPNet output shape: {output.shape}")
            print(f"  ALPNet output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
            print(f"  ALPNet output mean: {output.mean().item():.4f}")
            
            # Check each channel
            if output.dim() == 4 and output.shape[1] > 1:
                for c in range(output.shape[1]):
                    ch = output[0, c]
                    print(f"    Channel {c}: min={ch.min().item():.4f}, max={ch.max().item():.4f}, "
                          f"mean={ch.mean().item():.4f}, >0.5={((ch>0.5).sum().item())}/{ch.numel()}")
        
        # output shape: (1, 2, H, W) = [background_logits, foreground_logits]
        # Apply softmax to get probabilities
        if output.dim() == 4 and output.shape[1] > 1:
            probs = torch.softmax(output, dim=1)  # (1, 2, H, W)
            pred = probs[:, 1, :, :]  # foreground probability
        else:
            # If single channel, apply sigmoid
            pred = torch.sigmoid(output.squeeze(0).squeeze(0))
            if pred.dim() == 2:
                pass
            else:
                pred = pred.squeeze()
        
        # Return (H, W)
        return pred.squeeze(0).cpu()


# ============================================================================
# SAM Model Wrapper
# ============================================================================

# class SAMModel:
#     """SAM model wrapper"""
    
#     def __init__(self, config: ProtoSAMConfig):
#         self.config = config
#         self.model = None
#         self._init_model()
    
#     def _init_model(self):
#         """Initialize SAM model"""
#         from models.SamWrapper import SamWrapper
        
#         sam_args = {
#             "model_type": self.config.protosam_sam_ver,
#             "sam_checkpoint": self.config.sam_checkpoint,
#         }
        
#         self.model = SamWrapper(sam_args=sam_args).to(self.config.device)
#         self.model.eval()


# ============================================================================
# ProtoSAM Model (Combined ALPNet + SAM)
# ============================================================================

class ProtoSAMModel:
    """ProtoSAM model combining ALPNet with SAM refinement"""
    
    def __init__(self, config: ProtoSAMConfig):
        self.config = config
        self.alpnet = ALPNetModel(config)
        self._init_protosam()
        self.sam_trans = ResizeLongestSide(self.config.image_size[0])
    
    def _init_protosam(self):
        if self.config.coarse_pred_only:
            self.model = self.alpnet.model
        else:
            if self.config.protosam_sam_ver in ("sam_h", "sam_b"):
                print("  Loading ProtoSAM...")
                self.model = ProtoSAM(
                    image_size=self.config.image_size,
                    coarse_segmentation_model=self.alpnet.model,
                    use_bbox=self.config.use_bbox,
                    use_points=self.config.use_points,
                    use_mask=self.config.use_mask,
                    debug=self.config.debug,
                    num_points_for_sam=self.config.num_points_for_sam,
                    use_cca=self.config.do_cca,
                    point_mode=self.config.point_mode,
                    use_sam_trans=self.config.use_sam_trans,
                    coarse_pred_only=self.config.coarse_pred_only,
                    sam_pretrained_path=self.config.sam_checkpoint,
                    use_neg_points=self.config.use_neg_points,
                )
            elif self.config.protosam_sam_ver == "medsam":
                self.model = ProtoMedSAM(
                    image_size=self.config.image_size,
                    coarse_segmentation_model=self.alpnet.model,
                    debug=self.config.debug,
                    use_cca=self.config.do_cca,
                )
        
        self.model.to(self.config.device)
        self.model.eval()
        print("  ProtoSAM model loaded")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image: numpy (H,W,3) -> tensor (1,3,1024,1024)"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # SAM resize: returns (1024, 1024, 3)
        image = self.sam_trans.apply_image(image)
        
        # To tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # SAM normalization
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - pixel_mean) / pixel_std
        
        return image_tensor
    
    def _preprocess_mask(self, mask: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        """Preprocess mask: numpy (H,W) -> tensor (1, 1024, 1024)"""
        if mask is None:
            return None
        
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure values in [0, 1]
        if mask.max() > 1.0:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = (mask > 0.5).astype(np.uint8)
        
        # SAM resize
        mask = self.sam_trans.apply_image(mask)  # (1024, 1024)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, 1024, 1024)
        
        return mask_tensor
    
    def predict(self,
               query_image: np.ndarray,
               support_image: np.ndarray,
               support_mask: np.ndarray,
               query_gt: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Predict segmentation."""
        try:
            # Preprocess
            query_tensor = self._preprocess_image(query_image).to(self.config.device)
            support_img_tensor = self._preprocess_image(support_image).to(self.config.device)
            support_mask_tensor = self._preprocess_mask(support_mask)
            
            if support_mask_tensor is not None:
                support_mask_tensor = support_mask_tensor.to(self.config.device)
            else:
                print("  WARNING: support_mask is None")
                return None
            
            if self.config.debug:
                print(f"  Query shape: {query_tensor.shape}")
                print(f"  Support img shape: {support_img_tensor.shape}")
                print(f"  Support mask shape: {support_mask_tensor.shape}")
                print(f"  Support mask unique values: {support_mask_tensor.unique().cpu().tolist()}")
                print(f"  Support mask sum: {support_mask_tensor.sum().item()}")
            
            with torch.no_grad():
                if self.config.coarse_pred_only:
                    # ALPNet only
                    pred = self.alpnet.predict(
                        query_image=query_tensor.squeeze(0),
                        support_images=support_img_tensor.squeeze(0),
                        support_mask=support_mask_tensor.squeeze(0),
                        isval=True,
                        val_wsize=2,
                    )
                else:
                    # Build coarse input for ProtoSAM
                    coarse_input = {
                        'query_image': query_tensor,
                        'support_images': support_img_tensor,
                        'support_labels': support_mask_tensor,
                        'gts': self._preprocess_mask(query_gt).to(self.config.device) if query_gt is not None else None,
                        'isval': True,
                        'val_wsize': 2,
                        'original_sz': query_image.shape[:2],
                        'img_sz': query_image.shape[:2],
                    }
                    
                    # ProtoSAM forward
                    pred, scores = self.model(
                        query_images=query_tensor,  # This might expect a list
                        coarse_model_input=coarse_input,
                        degrees_rotate=0,
                    )
                    
                    # Handle output format
                    if isinstance(pred, list):
                        pred = pred[0]
            
            # Convert to numpy
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu()
                
                if self.config.debug:
                    print(f"  Raw pred shape: {pred.shape}")
                    print(f"  Raw pred min/max: {pred.min().item():.4f} / {pred.max().item():.4f}")
                
                # Squeeze batch dims
                while pred.dim() > 2:
                    pred = pred.squeeze(0)
                
                pred = pred.numpy()
            
            if self.config.debug:
                print(f"  Final pred shape: {pred.shape}")
                print(f"  Final pred min/max: {pred.min():.4f} / {pred.max():.4f}")
                print(f"  Final pred >0.5 ratio: {((pred > 0.5).sum()) / pred.size:.4f}")
            
            # Resize to original
            if pred.shape != query_image.shape[:2]:
                pred = cv2.resize(
                    pred,
                    (query_image.shape[1], query_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Binarize
            pred_binary = (pred > 0.5).astype(np.float32)
            
            return pred_binary
            
        except Exception as e:
            print(f"ProtoSAM predict error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_alpnet_input(self, query_tensor, support_img_tensor, support_mask_2d):
        """
        Build input dictionary for ProtoSAM's coarse_model_input.
        This must match the format expected by the original protosam function.
        """
        return {
            'query_image': query_tensor,                                    # (1, 3, 1024, 1024)
            'support_images': support_img_tensor.squeeze(0).unsqueeze(0),  # (1, 3, 1024, 1024)
            'support_labels': support_mask_2d.unsqueeze(0),                # (1, 1024, 1024)
            'gts': None,
            'isval': True,
            'val_wsize': 2,
            'original_sz': (1024, 1024),
            'img_sz': (1024, 1024),
        }
        


# ============================================================================
# Image-to-Image Matching Pipeline
# ============================================================================

class ImageToImageMatcher:
    """
    Image-to-image matching using ProtoSAM.
    Clean interface that replaces the messy protosam() function.
    """
    
    def __init__(self, config: ProtoSAMConfig):
        self.config = config
        self.model = None
        
        # 延迟初始化，避免在 import 时就加载模型
        print("  ProtoSAM model will be initialized on first use")
    
    def _ensure_model(self):
        """Lazy initialization of ProtoSAM model"""
        if self.model is None:
            self.model = ProtoSAMModel(self.config)
    
    def match_single(self,
                    query_image: np.ndarray,
                    support_image: np.ndarray,
                    support_mask: np.ndarray,
                    query_gt: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Single support image matching"""
        self._ensure_model()
        return self.model.predict(
            query_image=query_image,
            support_image=support_image,
            support_mask=support_mask,
            query_gt=query_gt,
        )
        
    def match(self,
             query_image: np.ndarray,
             support_images: List[np.ndarray],
             support_masks: List[np.ndarray],
             query_gt: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Match query image to support images.
        
        Args:
            query_image: (H, W, 3) RGB query image
            support_images: List of (H, W, 3) support images
            support_masks: List of (H, W) support masks
            query_gt: Optional ground truth mask
            
        Returns:
            Predicted mask (H, W), binary
        """
        if len(support_images) == 0:
            raise ValueError("No support images provided")
        
        # Use first support for 1-shot learning
        support_img = support_images[0]
        support_mask = support_masks[0]
        
        # Run prediction
        pred = self.model.predict(
            query_image=query_image,
            support_image=support_img,
            support_mask=support_mask,
            query_gt=query_gt,
        )
        
        return pred
    


# ============================================================================
# Evaluation Metrics (moved from alpnet_odE_tr_p0.py)
# ============================================================================

class SegmentationMetrics:
    """Compute segmentation metrics"""
    
    @staticmethod
    def dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
        """Dice coefficient"""
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)
        
        intersection = (pred_binary * gt_binary).sum()
        denom = pred_binary.sum() + gt_binary.sum()
        
        if denom == 0:
            return 1.0
        
        return 2.0 * intersection / (denom + smooth)
    
    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
        """IoU (Jaccard index)"""
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)
        
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        
        if union == 0:
            return 1.0
        
        return intersection / (union + smooth)
    
    @staticmethod
    def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
        """Precision and recall"""
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)
        
        tp = (pred_binary * gt_binary).sum()
        fp = (pred_binary * (1 - gt_binary)).sum()
        fn = ((1 - pred_binary) * gt_binary).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return float(precision), float(recall)
    
    @classmethod
    def compute_all(cls, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'dice': cls.dice(pred, gt),
            'iou': cls.iou(pred, gt),
            'precision': cls.precision_recall(pred, gt)[0],
            'recall': cls.precision_recall(pred, gt)[1],
        }


# ============================================================================
# Utility: set random seed
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Convenience function: protosam (replacement for original)
# ============================================================================

def protosam_predict(query_image: np.ndarray,
                     support_image: np.ndarray,
                     support_mask: np.ndarray,
                     config: Optional[ProtoSAMConfig] = None,
                     **kwargs) -> np.ndarray:
    """
    Convenience function for ProtoSAM prediction.
    Replaces the original protosam() function in alpnet_odE_tr_p0.py.
    
    Args:
        query_image: (H, W, 3) RGB query image
        support_image: (H, W, 3) RGB support image
        support_mask: (H, W) binary support mask
        config: ProtoSAMConfig (optional, created from kwargs if not provided)
        **kwargs: Additional config parameters
        
    Returns:
        Predicted mask (H, W), binary
    """
    if config is None:
        config = ProtoSAMConfig(**kwargs)
    
    set_seed(config.seed)
    
    matcher = ImageToImageMatcher(config)
    pred = matcher.match_single(
        query_image=query_image,
        support_image=support_image,
        support_mask=support_mask,
    )
    
    return pred