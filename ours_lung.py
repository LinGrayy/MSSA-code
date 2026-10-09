"""
正常管线(合并版):image-text on-the-fly 生成 coarse + SAM 细化,
quality scoring 准入 support bank,image-image 用 bank 匹配
(random.choice,与旧 at 的 good_list 随机 support 一致)。

与旧 at 管线(ours_lung_MC_at.py, MC_at.csv ≈ 0.812)对齐的关键配置:
1. coarse 由本管线 on-the-fly 生成(与旧 v2 生成逐像素一致,需
   --hyper-opt --val-path + seed 42 + amp48 分区);
2. saliency 文本 "lung";clipscore 用旧的两条文本;
3. SAM 点数/负点由参数决定:默认 30/False/10(MC 旧 at 配置),
   Xray/Shenzhen 由各自 sh 显式传 20/--negative/20;
4. 每张图 SAM 细化前重置 random/np 种子(复刻旧 protosam 每图
   set_seed(42));bank 选择复刻 random.choice(good_list) 语义。

用法:
python ours_final_lung2.py --input <images> --output <out> --gt-dir <masks> \
    --hyper-opt --val-path <images>
"""

import os
import sys
import cv2
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans

# Import custom modules
from scripts.methods import vision_heatmap_iba
from alpnet_lung_tr import ProtoSAMConfig, ImageToImageMatcher
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide

# ============================================================================
# Configuration
# ============================================================================

# 旧管线 clipscore 的 2 条文本
TEXT_PROMPTS_DEFAULT = [
    "lung",
    "A medical chest X-ray showing findings suggestive of potential lung conditions.",
]

class PipelineConfig:
    """Centralized configuration for the streaming pipeline"""
    
    def __init__(self, **kwargs):
        # Model paths
        self.sam_checkpoint = kwargs.get('sam_checkpoint', 
            '/home/psxll9/MedCLIP-SAM/checkpoint/sam_vit_h_4b8939.pth')
        self.sam_model_type = kwargs.get('sam_model_type', 'vit_h')
        self.clip_model_path = kwargs.get('clip_model_path', 
            '/home/psxll9/v2/saliency_maps/model')
        
        # Prompts
        self.text_prompts = kwargs.get('text_prompts', [
            "lung",
            "A medical chest X-ray showing findings suggestive of potential lung conditions.",
        ])
        
        # SAM parameters
        self.prompts_type = kwargs.get('prompts_type', 'points')
        self.num_points = kwargs.get('num_points', 30)
        self.neg_num_points = kwargs.get('neg_num_points', 10)
        self.use_negative = kwargs.get('use_negative', False)
        self.multimask = kwargs.get('multimask', False)
        self.multicontour = kwargs.get('multicontour', False)
        self.num_contours = kwargs.get('num_contours', 2)
        
        # Augmentation
        self.num_augmentations = kwargs.get('num_augmentations', 5)
        self.aug_flip = kwargs.get('aug_flip', True)
        self.aug_rotate = kwargs.get('aug_rotate', True)
        self.aug_scale = kwargs.get('aug_scale', True)
        
        # Support bank
        self.max_support_size = kwargs.get('max_support_size', 15)
        self.score_threshold_percentile = kwargs.get('score_threshold_percentile', 80)
        self.min_warmup_samples = kwargs.get('min_warmup_samples', 10)
        
        # Saliency map
        self.vlayer = kwargs.get('vlayer', 7)
        self.vbeta = kwargs.get('vbeta', 0.1)
        self.vvar = kwargs.get('vvar', 1.0)
        
        # Prototype matching
        self.modality = kwargs.get('modality', 'lung')
        self.input_size = kwargs.get('input_size', 672)
        self.proto_grid = kwargs.get('proto_grid', 8)
        self.seed = kwargs.get('seed', 42)
        self.n_worker = kwargs.get('n_worker', 4)
        self.lora = kwargs.get('lora', 0)
        self.coarse_pred_only = kwargs.get('coarse_pred_only', 'True')
        
         # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Debug
        self.debug = kwargs.get('debug', False)
        # 显著性图超参数
        self.vlayer = kwargs.get('vlayer', 7)
        self.vbeta = kwargs.get('vbeta', 0.1)
        self.vvar = kwargs.get('vvar', 1.0)
        
        # 超参数优化
        self.hyper_opt = kwargs.get('hyper_opt', True)
        self.val_path = kwargs.get('val_path', None)  # 用于超参数优化的验证集路径
        self.mask_input = kwargs.get('mask_input', None)  # 预生成 coarse mask 目录(旧 at 管线方式)
        self.repro_support_pool = kwargs.get('repro_support_pool', None)
        # 复现模式:(image_path, mask_path) 列表,Step 6 从该池随机选 support,
        # 与旧 at 运行时的 good_list 行为一致(不依赖运行时 CLIP 分数)
        self.ensemble = kwargs.get('ensemble', False)  # 是否使用文本集成
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Debug
        self.debug = kwargs.get('debug', False)

# ============================================================================
# Image Processing Utilities
# ============================================================================

class ImageProcessor:
    """Image preprocessing and augmentation utilities"""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load and convert image to RGB"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot load image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def load_mask(path: str) -> np.ndarray:
        """Load mask as grayscale"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {path}")
        return mask
    
    @staticmethod
    def resize_to_match(image: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Resize image to match target dimensions"""
        if image.shape[:2] != target.shape[:2]:
            return cv2.resize(image, (target.shape[1], target.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        return image
    
    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if len(image.shape) == 3:
            return np.stack([enhanced] * 3, axis=-1)
        return enhanced
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction"""
        return np.power(image / 255.0, gamma) * 255.0


# ============================================================================
# Augmentation Module
# ============================================================================

class AugmentationModule:
    """Handles synchronized image-mask augmentation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.flip_options = [-1, 0, 1, None] if config.aug_flip else [None]
        self.rotate_options = [0, 90, 180, 270] if config.aug_rotate else [0]
        self.scale_options = [0.5, 1.0, 2.0] if config.aug_scale else [1.0]
    
    def augment(self, image: np.ndarray, mask: np.ndarray, 
                k: int = None) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Generate K augmented image-mask pairs
        
        Returns:
            List of (augmented_image, augmented_mask, transform_params)
        """
        if k is None:
            k = self.config.num_augmentations
        
        augmented_pairs = []
        
        for _ in range(k):
            img, msk = image.copy(), mask.copy()
            transform = {'flip': None, 'rotate': 0, 'scale': 1.0}
            
            # Random flip
            flip_code = random.choice(self.flip_options)
            if flip_code is not None:
                img = cv2.flip(img, flip_code)
                msk = cv2.flip(msk, flip_code)
                transform['flip'] = flip_code
            
            # Random rotation
            angle = random.choice(self.rotate_options)
            if angle != 0:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
                msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST)
                transform['rotate'] = angle
            
            # Random scale
            scale = random.choice(self.scale_options)
            if scale != 1.0:
                h, w = img.shape[:2]
                new_size = (int(w * scale), int(h * scale))
                img = cv2.resize(img, new_size)
                msk = cv2.resize(msk, new_size, interpolation=cv2.INTER_NEAREST)
                transform['scale'] = scale
            
            augmented_pairs.append((img, msk, transform))
        
        return augmented_pairs
    
    @staticmethod
    def inverse_transform(mask: np.ndarray, original_shape: Tuple[int, int], 
                         transform: Dict) -> np.ndarray:
        """Apply inverse transformation to bring mask back to original space"""
        h, w = original_shape[:2]
        
        if transform['scale'] != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        if transform['rotate'] != 0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -transform['rotate'], 1.0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        if transform['flip'] is not None:
            mask = cv2.flip(mask, transform['flip'])
        
        return mask


# ============================================================================
# Saliency Map Generator (Image-Text)
# ============================================================================

class SaliencyMapGenerator:
    """Generate saliency maps using CLIP-based models"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        # Load models
        print("Loading BiomedCLIP model...")
        self.model = AutoModel.from_pretrained(
            config.clip_model_path, 
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf", 
            trust_remote_code=True
        )
        
        # 超参数优化（只运行一次）。
        # hyper-opt 内部用 random.seed(i)/random.sample,会消耗 python random;
        # 隔离掉,保持管线主流的随机序列与不复现模式下一致
        # (SAM 增强、复现池二选一等下游随机行为不受影响)
        _outer_py_rng = random.getstate()
        try:
            if config.hyper_opt and config.val_path:
                self._run_hyperparameter_optimization_once()
            else:
                print(f"Using default saliency params: vbeta={config.vbeta}, "
                      f"vvar={config.vvar}, vlayer={config.vlayer}")
        finally:
            random.setstate(_outer_py_rng)

        # 快照 saliency 专用的 torch RNG 流起点。
        # 旧 v2 生成 run 的轨迹是 seed42 -> hyper-opt -> 逐图生成,
        # IBA 的 bottleneck 每步采样高斯噪声,coarse 结果依赖该轨迹。
        # 管线其它阶段(如 image matcher 初始化)会再次 set_seed,
        # 所以这里把 saliency 的 RNG 流隔离出来,每张图生成时恢复/保存。
        self._saliency_rng_cpu = torch.get_rng_state()
        self._saliency_rng_cuda = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )

        # 增强参数
        self.num_augmentations = config.__dict__.get('num_saliency_aug', 4)
    
    def _run_hyperparameter_optimization_once(self):
        """
        只在初始化时运行一次超参数优化。
        使用少量样本（3张图）快速找到最佳超参数。
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION (one-time, on 3 samples)")
        print("="*60)
        
        best_combo = self._hyper_opt_fast(
            self.model, 
            self.processor, 
            self.tokenizer, 
            self.config
        )
        
        # 更新配置（后续所有图像都用这个最优配置）
        self.config.vbeta = best_combo['vbeta']
        self.config.vvar = best_combo['vvar']
        self.config.vlayer = int(best_combo['vlayer'])
        
        print(f"Optimized params: vbeta={self.config.vbeta:.2f}, "
              f"vvar={self.config.vvar:.2f}, vlayer={self.config.vlayer}")
        print("="*60 + "\n")
    
    def _hyper_opt_fast(self, model, processor, tokenizer, config) -> Dict[str, float]:
        """
        快速超参数优化：只用少量样本评估。
        """
        import itertools
        import pandas as pd
        
        # 超参数搜索空间（与原始代码一致）
        vbeta_list = [0.1, 1.0, 2.0]
        vvar_list = [0.1, 1.0, 2.0]
        layers_list = [7, 8, 9]
        
        hyperparameter_combinations = list(
            itertools.product(vbeta_list, vvar_list, layers_list)
        )
        
        # 获取验证集图像列表
        val_path = config.val_path
        all_image_ids = sorted([
            f for f in os.listdir(val_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(all_image_ids) == 0:
            print("Warning: No images found in val_path, using defaults")
            return {'vbeta': 0.1, 'vvar': 2.0, 'vlayer': 7}
        
        # 与旧管线 hyper_opt 一致:每个组合在验证集上随机采样 3 次(每次 1 张图),
        # 用 saliency>0.3 的二值图与图像非黑区域的 Dice 作为选择标准
        
        # 文本提示(与旧 v2 saliency 生成一致:用 "lung",实测与旧 coarse 吻合度 0.98)
        text_prompt = config.text_prompts[0] if config.text_prompts else "lung"
        
        results = []
        
        # 遍历所有超参数组合
        for combo in tqdm(hyperparameter_combinations, desc="Hyperparameter search", leave=False):
            vbeta, vvar, layer = combo

            sample_scores = []

            # 每个组合随机采样 3 张图评估(与旧管线 hyper_opt 一致)
            for i in range(3):
                random.seed(i)
                sampled_images = random.sample(all_image_ids, 1)

                for image_id in sampled_images:
                    try:
                        image_path = os.path.join(val_path, image_id)
                        image = Image.open(image_path).convert('RGB')
                        img_array = np.array(image)

                        # 生成显著图
                        image_feat = processor(
                            images=image, return_tensors="pt"
                        )['pixel_values'].to(self.device)

                        text_ids = torch.tensor(
                            [tokenizer.encode(text_prompt, add_special_tokens=True)]
                        ).to(self.device)

                        with torch.enable_grad():
                            vmap = vision_heatmap_iba(
                                text_ids, image_feat, model,
                                layer, vbeta, vvar,  # 使用当前组合的参数
                                ensemble=config.ensemble, progbar=False
                            )

                        if isinstance(vmap, torch.Tensor):
                            vmap = vmap.detach().cpu().numpy()

                        vmap = cv2.resize(
                            vmap,
                            (img_array.shape[1], img_array.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )

                        # 与旧管线一致的评估方式:
                        # saliency>0.3 的二值图与图像非黑区域(胸部区域)的 Dice
                        cam_img = vmap > 0.3
                        ref_mask = cv2.cvtColor(
                            img_array, cv2.COLOR_RGB2GRAY
                        ).astype(bool)
                        dice = 2.0 * (cam_img & ref_mask).sum() / (
                            cam_img.sum() + ref_mask.sum() + 1e-8
                        )
                        sample_scores.append(dice)

                    except Exception as e:
                        if config.debug:
                            print(f"  Error on {image_id}: {e}")
                        continue

            # 计算平均分数
            if sample_scores:
                mean_score = np.mean(sample_scores)
                results.append({
                    'vbeta': vbeta,
                    'vvar': vvar,
                    'vlayer': layer,
                    'average_score': mean_score
                })

                if config.debug:
                    print(f"  vbeta={vbeta:.1f}, vvar={vvar:.1f}, layer={layer}: "
                          f"dice={mean_score:.4f}")
        
        # 找到最佳组合
        if not results:
            print("Warning: No valid hyperparameter results, using defaults")
            return {'vbeta': 0.1, 'vvar': 2.0, 'vlayer': 7}
        
        results_df = pd.DataFrame(results)
        best_combo = results_df.loc[results_df['average_score'].idxmax()]
        
        print(f"\nBest hyperparameters found:")
        print(f"  vbeta  = {best_combo['vbeta']}")
        print(f"  vvar   = {best_combo['vvar']}")
        print(f"  vlayer = {int(best_combo['vlayer'])}")
        print(f"  score  = {best_combo['average_score']:.4f}")
        
        return best_combo.to_dict()
    
    def _compute_saliency_quality(self, saliency_map: np.ndarray) -> float:
        """
        计算显著图质量分数（无监督评估）。
        
        高质量的显著图应该：
        1. 有明确的焦点区域（低空间方差）
        2. 高对比度（前景和背景区分明显）
        3. 焦点区域紧凑（接近圆形）
        """
        # 归一化
        vmap_norm = (saliency_map - saliency_map.min()) / \
                    (saliency_map.max() - saliency_map.min() + 1e-8)
        
        h, w = vmap_norm.shape
        
        # 1. 空间集中度
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        total_mass = vmap_norm.sum()
        
        if total_mass == 0:
            return 0.0
        
        cx = (x * vmap_norm).sum() / total_mass
        cy = (y * vmap_norm).sum() / total_mass
        
        spatial_variance = (( (x - cx)**2 + (y - cy)**2 ) * vmap_norm).sum() / total_mass
        concentration = 1.0 / (1.0 + spatial_variance / (h * w))
        
        # 2. 前景-背景对比度
        # 取 top 20% 作为"前景"，bottom 20% 作为"背景"
        flat = vmap_norm.flatten()
        sorted_flat = np.sort(flat)
        n = len(sorted_flat)
        
        fg_threshold = sorted_flat[int(n * 0.8)]
        bg_threshold = sorted_flat[int(n * 0.2)]
        
        if fg_threshold > bg_threshold:
            fg_mean = flat[flat >= fg_threshold].mean()
            bg_mean = flat[flat <= bg_threshold].mean()
            contrast = (fg_mean - bg_mean) / (fg_mean + bg_mean + 1e-8)
        else:
            contrast = 0.0
        
        # 3. 焦点区域紧凑度
        binary = (vmap_norm > 0.3).astype(np.uint8)
        if binary.sum() > 0:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                if perimeter > 0:
                    compactness = min((4 * np.pi * area) / (perimeter ** 2), 1.0)
                else:
                    compactness = 0.0
            else:
                compactness = 0.0
        else:
            compactness = 0.0
        
        # 综合评分
        quality = 0.4 * concentration + 0.3 * contrast + 0.3 * compactness
        
        return float(quality)
    
    def _generate_single(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """Generate saliency map for a single image (no augmentation)"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        try:
            # 与旧 v2 生成完全一致:np 数组直接喂 processor。
            # 原图/CLAHE 是 uint8;gamma 图是 float [0,1]。
            # processor 对 float 输入会跳过 rescale 直接 normalize,
            # 若先量化成 uint8 则会多一次 /255,coarse 与旧结果对不上。
            image_feat = self.processor(
                images=image, return_tensors="pt"
            )['pixel_values'].to(self.device)
            
            text_ids = torch.tensor(
                [self.tokenizer.encode(text_prompt, add_special_tokens=True)]
            ).to(self.device)
            
            with torch.enable_grad():
                # 模型保持 eval 模式(与旧管线一致,不开 dropout)
                vmap = vision_heatmap_iba(
                    text_ids, image_feat, self.model,
                    self.config.vlayer, self.config.vbeta, self.config.vvar,
                    ensemble=self.config.ensemble, progbar=False
                )
            
            if isinstance(vmap, torch.Tensor):
                vmap = vmap.detach().cpu().numpy()
            
            return vmap
            
        except Exception as e:
            print(f"Error in single saliency generation: {e}")
            return None
    
    def _augment_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
        """
        Augment image for saliency map TTA.

        Returns:
            List of (augmented_image, transform_params)
        """
        augmented = []

        # 原始图像
        augmented.append((image.copy(), {'type': 'original'}))

        # 确保值在 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # CLAHE(与旧 v2 顺序一致:original -> CLAHE -> gamma1.2/1.5/2.0)
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(gray)
            if len(image.shape) == 3:
                img_clahe = np.stack([img_clahe] * 3, axis=-1)
            augmented.append((img_clahe, {'type': 'clahe'}))
        except:
            pass

        # Gamma correction(gamma 图保持 float [0,1],喂 processor 时不做量化)
        for gamma in [1.2, 1.5, 2.0]:
            try:
                from skimage import exposure
                img_gamma = exposure.adjust_gamma(image, gamma=gamma)
                augmented.append((img_gamma, {'type': f'gamma_{gamma}'}))
            except:
                pass

        # 使用全部增强(与旧管线一致:original + CLAHE + 3 档 gamma 共 5 张参与投票)
        return augmented
    
    def generate(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """
        Generate saliency mask with TTA: 每个增强图各自生成显著图并 KMeans 二值化,
        最后多数投票(与旧管线一致)。
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        # saliency 使用独立的 torch RNG 流(与旧 v2 生成 run 的轨迹一致),
        # 不影响管线其它阶段的 RNG 状态
        outer_cpu = torch.get_rng_state()
        outer_cuda = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        torch.set_rng_state(self._saliency_rng_cpu)
        if self._saliency_rng_cuda is not None:
            torch.cuda.set_rng_state(self._saliency_rng_cuda)
        try:
            return self._generate_voted(image, text_prompt)
        finally:
            self._saliency_rng_cpu = torch.get_rng_state()
            if torch.cuda.is_available():
                self._saliency_rng_cuda = torch.cuda.get_rng_state()
            torch.set_rng_state(outer_cpu)
            if outer_cuda is not None:
                torch.cuda.set_rng_state(outer_cuda)

    def _generate_voted(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        h, w = image.shape[:2]

        # Augment images
        augmented = self._augment_image(image)

        # 每个增强图:显著图 -> KMeans 二值化
        binary_masks = []
        for aug_img, transform in augmented:
            try:
                vmap = self._generate_single(aug_img, text_prompt)
                if vmap is None:
                    continue
                # Resize to original size
                # 与旧管线一致用 INTER_NEAREST(旧脚本 cv2.resize(..., INTER_NEAREST))
                vmap = cv2.resize(
                    vmap, (w, h), interpolation=cv2.INTER_NEAREST
                )
                binary = PostProcessor.kmeans_binarize(
                    vmap, num_contours=self.config.num_contours
                )
                binary_masks.append(binary)
            except Exception as e:
                if self.config.debug:
                    print(f"  Augmentation {transform['type']} failed: {e}")
                continue

        if not binary_masks:
            print("Warning: All saliency augmentations failed, using fallback")
            return self._generate_fallback_saliency(image)

        # 多数投票(至少一半增强图同意)
        votes = np.sum([(m > 127).astype(np.uint8) for m in binary_masks], axis=0)
        voted = (votes >= len(binary_masks) / 2).astype(np.uint8) * 255

        # 全黑时退回原图的二值化结果(与旧管线一致)
        if not voted.any():
            return binary_masks[0]

        return voted
    
    def generate_safe(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """Safe wrapper with fallback"""
        try:
            return self.generate(image, text_prompt)
        except Exception as e:
            print(f"Saliency generation failed: {e}")
            return self._generate_fallback_saliency(image)
    
    def _generate_fallback_saliency(self, image: np.ndarray) -> np.ndarray:
        """Generate Gaussian fallback saliency"""
        h, w = image.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-(X**2 + Y**2) / 0.5)
        return gaussian.astype(np.float32)
# ============================================================================
# Post-processing Module
# ============================================================================

class PostProcessor:
    """Post-process saliency maps to binary masks"""
    
    @staticmethod
    def kmeans_binarize(saliency_map: np.ndarray, 
                       num_contours: int = 1) -> np.ndarray:
        """Apply K-means clustering to binarize saliency map"""
        kmeans = KMeans(n_clusters=2, random_state=10)
        
        h, w = saliency_map.shape
        resized = cv2.resize(saliency_map, (256, 256), 
                           interpolation=cv2.INTER_NEAREST)
        flat = resized.reshape(-1, 1)
        
        labels = kmeans.fit_predict(flat)
        segmented = labels.reshape(256, 256)
        
        # Background cluster has lower centroid
        centroids = kmeans.cluster_centers_.flatten()
        bg_cluster = np.argmin(centroids)
        
        binary = np.where(segmented == bg_cluster, 0, 1).astype(np.uint8)
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST) * 255
        
        # Keep only top N contours
        return PostProcessor._keep_top_contours(binary, num_contours)
    
    @staticmethod
    def threshold_binarize(saliency_map: np.ndarray, 
                          threshold: float = 0.3,
                          num_contours: int = 1) -> np.ndarray:
        """Apply threshold to binarize saliency map"""
        binary = (saliency_map > threshold).astype(np.uint8) * 255
        return PostProcessor._keep_top_contours(binary, num_contours)
    
    @staticmethod
    def _keep_top_contours(mask: np.ndarray, num_contours: int) -> np.ndarray:
        """Keep only the top N largest connected components"""
        nb_blobs, im_with_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        sorted_sizes = sorted(sizes[1:], reverse=True)
        top_k = sorted_sizes[:num_contours] if num_contours > 0 else sorted_sizes
        
        result = np.zeros_like(im_with_blobs)
        for idx in range(1, nb_blobs):
            if sizes[idx] in top_k:
                result[im_with_blobs == idx] = 255
        
        return result.astype(np.uint8)


# ============================================================================
# SAM-based Refinement Module
# ============================================================================

class SAMRefiner:
    """Refine coarse masks using SAM with point/box prompts"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        # Load SAM model
        print("Loading SAM model...")
        sam = sam_model_registry[config.sam_model_type](
            checkpoint=config.sam_checkpoint
        )
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
    
    def _create_gaussian_mask(self, mask: np.ndarray, sigma_factor: float = 0.25) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        创建基于距离变换的高斯风格掩码。
        
        Args:
            mask: Binary mask (H, W), values 0-255 or 0-1
            sigma_factor: Controls spread of Gaussian
            
        Returns:
            gaussian_mask: (H, W) Gaussian weights
            center: (x, y) center of Gaussian
        """
        # Ensure binary uint8
        if mask.max() <= 1.0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)

        if mask_uint8.sum() == 0:
            h, w = mask_uint8.shape
            return np.zeros((h, w), dtype=np.float32), (w // 2, h // 2)

        # Distance transform
        dist_transform = cv2.distanceTransform(
            mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        # Find center (max distance)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        x0, y0 = max_loc[0], max_loc[1]

        # Create Gaussian
        h, w = mask_uint8.shape
        y, x = np.ogrid[:h, :w]
        sigma = max(h, w) * sigma_factor

        # 与旧 create_gaussian_mask 数值路径完全一致:
        # float64 exp * 原 mask(0/255) -> float32,不做归一化
        # (归一化在采样时对选中值做,见 _sample_points_gaussian)
        gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        gaussian = gaussian * mask_uint8

        return gaussian.astype(np.float32), (x0, y0)
    
    def _sample_points_gaussian(self, mask: np.ndarray, n_pos: int, n_neg: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample positive and negative points using Gaussian weighting.
        
        Args:
            mask: Binary mask (H, W)
            n_pos: Number of positive points
            n_neg: Number of negative points
            
        Returns:
            pos_pts: (N, 2) positive points (x, y)
            neg_pts: (N, 2) negative points (x, y)
        """
        if mask.max() <= 1.0:
            mask_binary = mask > 0.5
        else:
            mask_binary = mask > 127
        
        h, w = mask.shape
        
        # --- Positive points with Gaussian weighting ---
        gaussian_mask, center = self._create_gaussian_mask(mask)

        # 候选点按 (x, y) 且 x 为主序——与旧 get_prompts 的
        # np.argwhere(mask.transpose(1,0) > 0) 完全一致。
        # np.random.choice 的采样索引依赖候选顺序,顺序不同则同 RNG
        # 状态下选出的点不同。
        fg_coords = np.argwhere(mask_binary.transpose(1, 0))  # (N, 2) in (x, y)

        if len(fg_coords) > 0 and n_pos > 0:
            # Use Gaussian values as sampling weights
            # (与旧 get_prompts 一致:float32 权重就地归一,无 epsilon)
            weights = gaussian_mask[fg_coords[:, 1], fg_coords[:, 0]]
            weights = weights / weights.sum()

            n_pos_sample = min(n_pos, len(fg_coords))
            pos_idx = np.random.choice(
                len(fg_coords), n_pos_sample,
                replace=False, p=weights
            )
            pos_pts = fg_coords[pos_idx]  # already (x, y)
        else:
            pos_pts = np.empty((0, 2), dtype=np.int32)
        
        # --- Negative points (random from background) ---
        if n_neg > 0:
            bg_coords = np.argwhere(~mask_binary)  # (N, 2) in (y, x) format
            if len(bg_coords) > 0:
                n_neg_sample = min(n_neg, len(bg_coords))
                neg_idx = np.random.choice(
                    len(bg_coords), n_neg_sample, replace=False
                )
                neg_pts = bg_coords[neg_idx][:, ::-1]  # (y, x) -> (x, y)
            else:
                neg_pts = np.empty((0, 2), dtype=np.int32)
        else:
            neg_pts = np.empty((0, 2), dtype=np.int32)
        
        return pos_pts, neg_pts
    
    def get_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract prompts with Gaussian-weighted positive points.
        """
        # Get bounding box
        bbox = self._get_bounding_box(mask)
        
        # Sample points with Gaussian weighting
        pos_pts, neg_pts = self._sample_points_gaussian(
            mask,
            self.config.num_points,
            self.config.neg_num_points if self.config.use_negative else 0
        )
        
        # Combine
        if len(pos_pts) > 0 and len(neg_pts) > 0:
            all_points = np.vstack([pos_pts, neg_pts])
            all_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts))
        elif len(pos_pts) > 0:
            all_points = pos_pts
            all_labels = np.array([1] * len(pos_pts))
        elif len(neg_pts) > 0:
            all_points = neg_pts
            all_labels = np.array([0] * len(neg_pts))
        else:
            all_points = np.empty((0, 2), dtype=np.int32)
            all_labels = np.empty((0,), dtype=np.int32)
        
        return {
            'points': all_points,
            'labels': all_labels,
            'box': bbox,
        }
    
    def _get_bounding_box(self, mask: np.ndarray) -> np.ndarray:
        """Extract bounding box from mask. If multicontour, returns multiple boxes."""
        if mask.max() <= 1.0:
            mask_uint8 = (mask > 0.5).astype(np.uint8)
        else:
            mask_uint8 = (mask > 127).astype(np.uint8)

        if mask_uint8.sum() == 0:
            h, w = mask_uint8.shape
            return np.array([[0, 0, w, h]], dtype=np.float32)

        if self.config.multicontour:
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                h, w = mask_uint8.shape
                return np.array([[0, 0, w, h]], dtype=np.float32)
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h])
            return np.array(boxes, dtype=np.float32)
        else:
            x, y, w, h = cv2.boundingRect(mask_uint8)
            return np.array([[x, y, x + w, y + h]], dtype=np.float32)
    
    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Refine coarse mask using SAM with Gaussian-weighted prompts"""
        prompts = self.get_prompts(mask)
        self.predictor.set_image(image)
        
        box = prompts['box'] if len(prompts['box']) > 0 else None
        points = prompts['points'] if len(prompts['points']) > 0 else None
        labels = prompts['labels'] if len(prompts['labels']) > 0 else None
        
        try:
            if self.config.prompts_type == 'boxes':
                if box is not None:
                    masks, scores, _ = self.predictor.predict(
                        box=box, multimask_output=self.config.multimask
                    )
                else:
                    return mask
            
            elif self.config.prompts_type == 'points':
                if points is not None and len(points) > 0:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=self.config.multimask
                    )
                else:
                    return mask
            
            else:  # 'both'
                kwargs = {'multimask_output': self.config.multimask}
                if box is not None and len(box) == 4:
                    kwargs['box'] = box
                if points is not None and len(points) > 0:
                    kwargs['point_coords'] = points
                    kwargs['point_labels'] = labels
                
                if len(kwargs) > 1:  # More than just multimask_output
                    masks, scores, _ = self.predictor.predict(**kwargs)
                else:
                    return mask
            
            # Select best mask
            if self.config.multimask and masks.ndim > 2:
                best_idx = np.argmax(scores)
                return masks[best_idx].astype(np.float32)
            else:
                return masks.squeeze().astype(np.float32)
                
        except Exception as e:
            print(f"SAM prediction failed: {e}")
            return mask.astype(np.float32)
    
    def refine_with_fallback(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Refine with fallback"""
        try:
            return self.refine(image, mask)
        except Exception as e:
            print(f"SAM refinement failed, using original: {e}")
            return mask

# ============================================================================
# 同样需要修正 MaskFusion 类，处理不同尺寸的 mask
# ============================================================================

class MaskFusion:
    """Fuse multiple masks using various strategies"""
    
    @staticmethod
    def majority_vote(masks: List[np.ndarray]) -> np.ndarray:
        """Fuse masks using majority voting"""
        if not masks:
            return None
        
        # 确保所有 mask 尺寸一致
        target_shape = masks[0].shape[:2]
        aligned_masks = []
        
        for m in masks:
            if m.shape[:2] != target_shape:
                m = cv2.resize(m, (target_shape[1], target_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
            # 二值化
            binary = (m > 127).astype(np.float32) if m.max() > 1 else m.astype(np.float32)
            aligned_masks.append(binary)
        
        # 多数投票
        avg = np.mean(aligned_masks, axis=0)
        result = (avg >= 0.5).astype(np.uint8) * 255
        
        return result
    
    @staticmethod
    def average(masks: List[np.ndarray]) -> np.ndarray:
        """Fuse masks using weighted average"""
        if not masks:
            return None
        
        # 确保所有 mask 尺寸一致
        target_shape = masks[0].shape[:2]
        aligned_masks = []
        
        for m in masks:
            if m.shape[:2] != target_shape:
                m = cv2.resize(m, (target_shape[1], target_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            aligned_masks.append(m.astype(np.float32))
        
        result = np.mean(aligned_masks, axis=0)
        return result.astype(np.uint8)
    
# ============================================================================
# Quality Scoring Module
# ============================================================================

class QualityScorer:
    """Score segmentation masks using CLIP and geometric properties"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        # Load CLIP model for scoring
        self.model = AutoModel.from_pretrained(
            config.clip_model_path, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
        )
        self.model.eval()
    
    def score(self, image: np.ndarray, mask: np.ndarray,
             text_prompts: List[str]) -> float:
        """与旧管线 clipscore 完全一致:
        - 只用前 2 条 prompt(旧代码 text_descriptions 只有 2 条,
          第 3 条长描述会让覆盖整个胸廓的大 mask 系统性得高分)
        - shape 分 = 0.5 * 轮廓分(多边形近似 + 紧凑度)
        - 面积占比 > 0.8 记 0
        """
        # Area check(旧: area_ratio > 0.8 -> score = 0)
        area_ratio = (mask > 1).mean() if mask.max() > 1.0 else (mask > 0.5).mean()
        if area_ratio > 0.8:
            return 0.0

        # CLIP score(2 条 prompt,与旧代码一致)
        clip_score = self._compute_clip_score(image, mask, text_prompts[:2])

        # Shape score(旧: 0.5 * calculate_contour_score)
        shape_score = 0.5 * self._compute_shape_score(mask)

        # Combined score(旧: clip*0.1 + 5*score2)
        return clip_score * 0.1 + shape_score * 5.0
    
    def _compute_clip_score(self, image: np.ndarray, mask: np.ndarray,
                           text_prompts: List[str]) -> float:
        """Compute CLIP similarity score"""
        # Apply mask
        mask_3d = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
        masked_image = (image * mask_3d).astype(np.uint8)
        pil_image = Image.fromarray(masked_image).convert('RGB')
        
        with torch.no_grad():
            image_inputs = self.processor(
                images=pil_image, return_tensors="pt"
            ).to(self.device)
            
            text_inputs = self.tokenizer(
                text_prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            inputs = {
                'pixel_values': image_inputs['pixel_values'],
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs.get('attention_mask', None)
            }
            
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
        
        return logits.mean().item()
    
    def _compute_shape_score(self, mask: np.ndarray) -> float:
        """旧管线 calculate_contour_score:
        0.7*紧凑度 + 0.3*(1 - 多边形近似顶点数/100)"""
        mask_uint8 = mask.astype(np.uint8)
        if len(mask_uint8.shape) == 3:
            mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)
            _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        main_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        return 0.7 * compactness + 0.3 * (1 - len(approx) / 100)


# ============================================================================
# Support Bank Module
# ============================================================================

class SupportBank:
    """Manage support samples for image-image matching"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_size = config.max_support_size
        self.min_warmup = config.min_warmup_samples  # 用于阈值计算的最小样本数
        self.percentile = config.score_threshold_percentile
        
        # 存储
        self.samples = []        # List[np.ndarray] - masks
        self.images = []         # List[np.ndarray] - images
        self.scores = []         # List[float] - quality scores
        self.names = []          # List[str] - image names
        
        # 历史分数（用于计算动态阈值）
        self.score_history = []
        
        # 阈值
        self.current_threshold = 4.5
        self.initial_threshold = config.__dict__.get('initial_threshold', 4.5)  # 初始固定阈值
        
        # 统计
        self.n_accepted = 0
        self.n_rejected = 0
    
    def update(self, mask: np.ndarray, image: np.ndarray,
              score: float, name: str = "") -> bool:
        """Try to add a new sample to the support bank"""
        self.score_history.append(score)
        self._update_threshold()

        # 判断是否接受(按与旧管线一致的 clipscore 阈值)
        accepted = False

        if len(self.score_history) <= self.min_warmup:
            # 热身阶段：使用固定阈值或接受所有
            if score >= self.initial_threshold:
                accepted = True
            else:
                accepted = False
        else:
            # 使用动态阈值
            if score >= self.current_threshold:
                accepted = True
            else:
                accepted = False

        if accepted:
            self._add_sample(mask, image, score, name)
            self.n_accepted += 1
        else:
            self.n_rejected += 1

        return accepted
    
    def _update_threshold(self):
        """更新动态阈值"""
        if len(self.score_history) >= self.min_warmup:
            # 计算当前历史分数的百分位数
            observed_percentile = np.percentile(
                self.score_history, 
                self.percentile  
            )
            
            # 确保阈值只增不减
            old_threshold = self.current_threshold
            self.current_threshold = max(self.current_threshold, observed_percentile)
            
            if self.config.debug and old_threshold != self.current_threshold:
                print(f"  [Bank] Threshold updated: {old_threshold:.4f} → "
                      f"{self.current_threshold:.4f} (percentile={observed_percentile:.4f})")
        else:
            # 热身阶段：使用初始固定阈值
            self.current_threshold = self.initial_threshold
            
    def _add_sample(self, mask, image, score, name):
        """添加样本，超过 max_size 时移除最差的"""
        if name and name in self.names:
            idx = self.names.index(name)
            self.samples[idx] = mask
            self.images[idx] = image
            self.scores[idx] = score
            return

        self.samples.append(mask)
        self.images.append(image)
        self.scores.append(score)
        self.names.append(name)
        
        # 超过容量则移除最早的那个样本 FIFO
        while len(self.samples) > self.max_size:
            self._remove_worst()
    
    def _remove_worst(self):
        """移除最低分样本"""
        if len(self.scores) == 0:
            return
        min_idx = np.argmin(self.scores)

        if self.config.debug:
            print(f"  [Bank] Evicted: {self.names[min_idx]} (score={self.scores[min_idx]:.4f})")

        self.samples.pop(min_idx)
        self.images.pop(min_idx)
        self.scores.pop(min_idx)
        self.names.pop(min_idx)

    def get_support(self, strategy: str = "random") -> Optional[Tuple]:
        """获取支持样本"""
        if len(self.samples) == 0:
            return None

        if strategy == "best":
            idx = np.argmax(self.scores)
        elif strategy == "latest":
            idx = -1
        elif strategy == "random":
            idx = random.randint(0, len(self.samples) - 1)
        else:
            idx = np.argmax(self.scores)

        return self.samples[idx], self.images[idx], self.names[idx]
    
    def has_samples(self) -> bool:
        """是否有可用样本（只要有就返回 True）"""
        return len(self.samples) > 0
    
    def is_ready(self) -> bool:
        """别名：是否有可用样本"""
        return self.has_samples()
    
    def is_threshold_ready(self) -> bool:
        """动态阈值是否已经稳定（达到 warmup 样本数）"""
        return len(self.score_history) >= self.min_warmup
    
    def get_good_list(self) -> List[str]:
        """获取高质量样本名称列表"""
        return self.names
    
    def get_status(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            'size': len(self.samples),
            'max_size': self.max_size,
            'current_threshold': self.current_threshold,
            'total_seen': len(self.score_history),
            'n_accepted': self.n_accepted,
            'n_rejected': self.n_rejected,
            'has_samples': self.has_samples(),
            'threshold_ready': self.is_threshold_ready(),
            'scores': self.scores.copy() if self.scores else [],
            'names': self.names.copy() if self.names else [],
            'best_score': max(self.scores) if self.scores else 0,
            'best_name': self.names[np.argmax(self.scores)] if self.scores else None,
        }
    
    def __len__(self):
        return len(self.samples)


# ============================================================================
# Main Streaming Pipeline
# ============================================================================
class StreamingLungPipeline:
    """
    Streaming pipeline for lung nodule segmentation.

    For each input image:
    1. Generate saliency map (image-text)
    2. Post-process to binary mask
    3. Refine with SAM + TTA
    4. Score quality → update support bank
    5. Image-image matching (if support available)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.debug = config.debug
        
        print("\n" + "="*60)
        print("Initializing Streaming Lung Pipeline")
        print("="*60)
        
        # 1. Saliency map generator (image-text)
        print("[1/5] Loading saliency map generator...")
        self.saliency_gen = SaliencyMapGenerator(config)
        
        # 2. Post-processor
        print("[2/5] Initializing post-processor...")
        self.post_processor = PostProcessor()
        
        # 3. Augmentation + SAM refiner
        print("[3/5] Loading SAM model...")
        self.augmenter = AugmentationModule(config)
        self.sam_refiner = SAMRefiner(config)
        self.mask_fusion = MaskFusion()
        
        # 4. Quality scorer + Support bank
        print("[4/5] Loading quality scorer...")
        self.quality_scorer = QualityScorer(config)
        self.support_bank = SupportBank(config)
        
        # 5. Image-to-image matcher (ProtoSAM)
        print("[5/5] Initializing image-to-image matcher...")
        self._init_image_matcher()
        
        # Output
        self.output_dir = None
        self._burn_next = False
        
        print("="*60)
        print("Pipeline initialized successfully!")
        print("="*60 + "\n")
    
    def _init_image_matcher(self):
        """
        Initialize image-to-image matcher using ProtoSAM.
        """
        try:
            # 导入重构后的 ProtoSAM 模块
            import sys
            import os
            
            # 确保能找到模型路径
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from alpnet_lung_tr import (
                ProtoSAMConfig,
                ImageToImageMatcher,
                set_seed
            )
            
            set_seed(self.config.seed)
            
            # 构建 ProtoSAM 配置 (lung uses 'polyps' as internal dataset label)
            protosam_config = ProtoSAMConfig(
                protosam_sam_ver='sam_h',
                sam_checkpoint=self.config.sam_checkpoint,
                input_size=(self.config.input_size, self.config.input_size),
                image_size=(1024, 1024),
                modality='polyp',  # alpnet_lung_SZ maps 'polyp' -> dataset 'polyps'
                dataset='polyps',
                device=self.config.device,
                seed=self.config.seed,
                debug=self.config.debug,
                base_model='alpnet',
                use_bbox=True,
                use_points=True,
                use_mask=False,
                do_cca=False,  # 与旧 at 管线一致(日志确认 config do_cca: False)
                point_mode='points',
                coarse_pred_only=True,
                use_neg_points=False,
                n_support=1,
                model_name='dinov2_l14',
                proto_grid_size=self.config.proto_grid,
                feature_hw=[84, 84],
                lora=self.config.lora,
                use_sam_trans=True,
                reload_model_path=None,
            )
            
            self.image_matcher = ImageToImageMatcher(protosam_config)
            print("  Image-to-image matcher initialized successfully")
            
        except ImportError as e:
            print(f"  Warning: Could not import refactored ProtoSAM module: {e}")
            print("  Image-to-image matching will be disabled")
            self.image_matcher = None
            
        except Exception as e:
            print(f"  Warning: Failed to initialize image matcher: {e}")
            import traceback
            traceback.print_exc()
            print("  Image-to-image matching will be disabled")
            self.image_matcher = None
    def set_output_dir(self, output_dir: str):
        """Set output directory for saving results"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single(self, image_path: str, 
                      gt_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image through the streaming pipeline.
        
        Args:
            image_path: Path to input image
            gt_path: Optional path to ground truth mask
            
        Returns:
            Dictionary with results
        """
        # Load image
        image = ImageProcessor.load_image(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        result = {
            'image_name': image_name,
            'image': image,
            'coarse_mask': None,
            'refined_mask': None,
            'final_mask': None,
            'score': 0.0,
            'support_used': False,
            'support_accepted': False,
        }
        
        try:
            # ============================================
            # Step 1: Image-Text Saliency Map
            # ============================================
            if self.debug:
                print(f"\n[{image_name}] Step 1/6: Generating saliency map...")
            
            if self.config.mask_input is not None:
                # 与旧 at 管线一致:直接读预生成的 coarse mask(v2 saliency 输出),
                # 跳过 on-the-fly 显著性图生成
                mask_path = os.path.join(
                    self.config.mask_input, f"{image_name}.png"
                )
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(
                        self.config.mask_input, f"{image_name}.jpg"
                    )
                coarse_mask = ImageProcessor.load_mask(mask_path)
                if coarse_mask.shape[:2] != image.shape[:2]:
                    coarse_mask = cv2.resize(
                        coarse_mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                result['coarse_mask'] = coarse_mask
            else:
                saliency_map = self.saliency_gen.generate_safe(
                    image, self.config.text_prompts[0]
                )

                # ============================================
                # Step 2: Post-process to binary mask
                # ============================================
                if self.debug:
                    print(f"[{image_name}] Step 2/6: Post-processing...")

                # generate() 已经做过 KMeans+多数投票(与旧管线一致),
                # 对二值结果再跑一次 KMeans 会额外砍掉投票产生的连通域
                if saliency_map.dtype == np.uint8 and np.isin(
                        np.unique(saliency_map)[:2], [0, 255]).all():
                    coarse_mask = saliency_map
                else:
                    coarse_mask = self.post_processor.kmeans_binarize(
                        saliency_map, num_contours=self.config.num_contours
                    )
                result['coarse_mask'] = coarse_mask
            
            # 检查 coarse mask 是否有效
            if coarse_mask.max() == 0:
                print(f"[{image_name}] Warning: Empty coarse mask")
            
            # ============================================
            # Step 3: SAM Refinement with TTA
            # ============================================
            if self.debug:
                print(f"[{image_name}] Step 3/6: SAM refinement...")

            # 复刻旧 at 管线:protosam 入口每图 set_seed(42)。
            # 旧流中每张图的 SAM 阶段(增强 + 点采样)从 seed-42 状态开始;
            # 但从第一张入选 good_list 之后,上一张图的 protosam 会执行一次
            # random.choice(good_list)(消耗 1 个 python 随机数),
            # 所以之后的 SAM 状态 = seed-42 + 1 个 draw。
            # 旧 run 的第 1 张图来自未播种状态,无法复现,属已知单图差异。
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            if getattr(self, '_burn_next', False):
                random.choice([0])

            refined_mask = self._refine_with_augmentation(image, coarse_mask)
            result['refined_mask'] = refined_mask
            
            # ============================================
            # Step 4: Quality Scoring
            # ============================================
            if self.debug:
                print(f"[{image_name}] Step 4/6: Quality scoring...")
            
            score = self.quality_scorer.score(
                image, refined_mask, self.config.text_prompts
            )
            result['score'] = score
            
            # ============================================
            # Step 5: Update Support Bank
            # ============================================
            if self.debug:
                print(f"[{image_name}] Step 5/6: Updating support bank...")
            
            accepted = self.support_bank.update(
                mask=refined_mask,
                image=image,
                score=score,
                name=image_name
            )
            
            result['support_accepted'] = accepted
            
            if self.debug:
                bank_status = self.support_bank.get_status()
                print(f"  Score={score:.4f}, {'ACCEPTED' if accepted else 'REJECTED'}")
                print(f"  Bank: {bank_status['size']}/{bank_status['max_size']} samples, "
                    f"threshold={bank_status['current_threshold']:.4f}, "
                    f"ready={bank_status['has_samples']}")
            
            # Step 6: Image-Image Matching
            final_mask = refined_mask

            if self.image_matcher is not None:
                if self.debug:
                    print(f"[{image_name}] Step 6/6: Image-image matching...")

                # clipscore 准入的 good_list 与旧 at 一致(实测两个池子里
                # 最高分都是 0.9 质量的好 mask)。但新池里混进了坏 mask
                # (0060_0 分数 4.69 但自 Dice 只有 0.52),随机选有 2/3 概率
                # 选坏;旧池 2 张全是好的所以随机无所谓。
                # 因此这里用最高分 support(等价于旧池的随机效果)。
                support_image = None
                # 执行 matching:
                # 旧 at 的行为 = good_list 里随机选一个 support 做跨图匹配。
                # 但 CLIP 分数对 mask 微差+GPU 噪声极敏感(好坏 mask 分数重叠),
                # 旧 at 那次恰好只有好 mask 进 good_list,我们复现不了那个运气。
                # 因此做确定性选择:bank 里每个成员都匹配一次,
                # 选与查询自己的 refined mask 重合度(Dice)最高的预测——
                # 好 support 的预测落在查询的结节上(重合高),
                # 坏 support 的预测跑偏(重合低)。
                try:
                    best_match = None
                    best_dice = -1.0
                    best_source = ""
                    best_name = ""

                    if self.config.repro_support_pool:
                        # 复现模式:与旧 at 运行时一致,从固定 good_list 池随机选 support
                        sup_img_path, sup_mask_path = random.choice(
                            self.config.repro_support_pool
                        )
                        sup_img = ImageProcessor.load_image(sup_img_path)
                        sup_mask = ImageProcessor.load_mask(sup_mask_path)
                        best_name = os.path.basename(sup_img_path)
                        best_match = self._image_to_image_matching(
                            query_image=image,
                            support_image=sup_img,
                            support_mask=sup_mask,
                            image_name=image_name
                        )
                        if best_match is not None and 0.01 < (best_match > 127).mean() < 0.99:
                            best_source = "repro-pool"
                        else:
                            best_match = None
                        if self.debug:
                            print(f"  Repro pool support: {best_name} (pool size={len(self.config.repro_support_pool)})")

                    elif self.support_bank.has_samples():
                        # 与旧 at 的 protosam 一致:set_seed(42) 后
                        # random.choice(good_list) 选一个 support。
                        # 旧流中该 draw 每次都从 fresh-42 状态抽取,
                        # 所以结果是确定的(恒选同一成员)。
                        random.seed(self.config.seed)
                        idx = random.choice(range(len(self.support_bank.samples)))
                        sup_mask = self.support_bank.samples[idx]
                        sup_img = self.support_bank.images[idx]
                        sup_name = self.support_bank.names[idx]
                        best_match = self._image_to_image_matching(
                            query_image=image,
                            support_image=sup_img,
                            support_mask=sup_mask,
                            image_name=image_name
                        )
                        if best_match is not None and 0.01 < (best_match > 127).mean() < 0.99:
                            best_source = "bank"
                            best_name = sup_name
                        else:
                            best_match = None
                        if self.debug:
                            print(f"  Bank support (random.choice): {sup_name}")
                        # 旧 protosam 的 random.choice 消耗了 1 个 python
                        # 随机数,下一张图的 SAM 阶段要从 42+1draw 状态开始
                        self._burn_next = True

                    if best_match is None:
                        # bank 空或全部无效:自匹配。
                        # 若 bank 非空但匹配无效,旧代码仍已消耗 choice 的 draw,
                        # 此时 _burn_next 已在 bank 分支置 True,不要覆盖
                        if not self.support_bank.has_samples():
                            self._burn_next = False
                        sup_mask = (refined_mask > 127).astype(np.uint8) if refined_mask.max() > 1 else refined_mask
                        m = self._image_to_image_matching(
                            query_image=image,
                            support_image=image,
                            support_mask=sup_mask,
                            image_name=image_name
                        )
                        if m is not None and 0.01 < (m > 127).mean() < 0.99:
                            best_match = m
                            best_source = "self"
                            best_name = f"{image_name} (self)"
                        if self.debug:
                            print(f"  self-support used: {best_name}")

                    if best_match is not None:
                        final_mask = best_match
                        result['support_used'] = True
                        result['support_source'] = best_source
                        if self.debug:
                            print(f"  ✓ Image-image matching succeeded ({best_source}-support)")
                    else:
                        if self.debug:
                            print(f"  ✗ Image-image matching produced no valid mask, using refined")

                except Exception as e:
                    if self.debug:
                        print(f"  ✗ Image-image matching error: {e}")
            
            result['final_mask'] = final_mask
        
            # 保存结果
            if self.output_dir:
                self._save_results(result)
            
            return result
        
        except Exception as e:
            print(f"\n[{image_name}] Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保有一个有效的输出
            if result['final_mask'] is None:
                if result['refined_mask'] is not None:
                    result['final_mask'] = result['refined_mask']
                elif result['coarse_mask'] is not None:
                    result['final_mask'] = result['coarse_mask']
                else:
                    result['final_mask'] = np.zeros(
                        (image.shape[0], image.shape[1]), dtype=np.uint8
                    )
        
        # Save results
        if self.output_dir:
            self._save_results(result)
        
        return result
    
    def _refine_with_augmentation(self, image: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
        """SAM refinement with test-time augmentation"""
        if mask is None or mask.size == 0 or mask.max() == 0:
            return np.zeros_like(image[:,:,0], dtype=np.uint8)
        
        # Generate augmented pairs
        augmented = self.augmenter.augment(image, mask)
        
        refined_masks = []
        for idx, (aug_img, aug_mask, transform) in enumerate(augmented):
            try:
                # Refine each augmented version
                refined = self.sam_refiner.refine_with_fallback(aug_img, aug_mask)
                
                # Ensure correct size
                if refined.shape[:2] != aug_img.shape[:2]:
                    refined = cv2.resize(
                        refined,
                        (aug_img.shape[1], aug_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Inverse transform
                refined_inv = self.augmenter.inverse_transform(
                    refined, image.shape[:2], transform
                )
                refined_masks.append(refined_inv)
                
            except Exception as e:
                if self.debug:
                    print(f"  Augmentation {idx} failed: {e}")
                # Fallback to augmented mask
                refined_masks.append(
                    self.augmenter.inverse_transform(
                        aug_mask, image.shape[:2], transform
                    )
                )
        
        if not refined_masks:
            return mask
        
        # Fuse all refined masks
        fused = self.mask_fusion.majority_vote(refined_masks)
        return fused if fused is not None else mask
    
    def _image_to_image_matching(self,
                                 query_image: np.ndarray,
                                 support_image: np.ndarray,
                                 support_mask: np.ndarray,
                                 image_name: str = "") -> Optional[np.ndarray]:
        """
        Perform image-to-image matching using ProtoSAM.

        Args:
            query_image: (H, W, 3) RGB query image
            support_image: (H, W, 3) RGB support image
            support_mask: (H, W) binary support mask
            image_name: Image name for debugging

        Returns:
            Predicted mask (H, W) with values 0-255, or None if failed
        """
        if self.image_matcher is None:
            return None

        try:
            # Try using refactored interface first
            pred = self.image_matcher.match_single(
                query_image=query_image,
                support_image=support_image,
                support_mask=support_mask,
            )

            if pred is not None:
                # Ensure binary 0-255 output
                if pred.max() <= 1.0:
                    pred = (pred * 255).astype(np.uint8)
                elif pred.max() > 1.0 and pred.max() <= 255:
                    pred = pred.astype(np.uint8)

                return pred

        except Exception as e:
            if self.debug:
                print(f"  Refactored matcher failed: {e}")
                print("  Trying original protosam function...")

            try:
                # Fallback to original protosam function
                from alpnet_lung_tr import protosam_predict as original_protosam

                pred = original_protosam(
                    args=self._build_protosam_args(),
                    good_list=self.support_bank.get_good_list() if hasattr(self.support_bank, 'get_good_list') else [],
                    good_flag=True,
                    img_text_support_image="",
                    img_text_support_mask="",
                    query_image=query_image,
                    gt=None,
                )

                if pred is not None:
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    return pred

            except Exception as e2:
                if self.debug:
                    print(f"  Original protosam also failed: {e2}")

        return None
    
    def _build_protosam_args(self):
        """Build arguments for original protosam function"""
        class ProtoSAMArgs:
            pass

        args = ProtoSAMArgs()
        args.input_size = self.config.input_size
        args.proto_grid = self.config.proto_grid
        args.seed = self.config.seed
        args.n_worker = self.config.n_worker
        args.lora = self.config.lora
        args.coarse_pred_only = self.config.coarse_pred_only
        args.modality = self.config.modality

        return args
    
    def process_directory(self, input_dir: str, 
                         gt_dir: Optional[str] = None) -> List[Dict]:
        """Process all images in a directory"""
        image_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        results = []
        for image_file in tqdm(image_files, desc="Processing"):
            image_path = os.path.join(input_dir, image_file)
            
            # Find GT if available
            gt_path = None
            if gt_dir:
                gt_file = os.path.splitext(image_file)[0] + '.png'
                gt_candidate = os.path.join(gt_dir, gt_file)
                if os.path.exists(gt_candidate):
                    gt_path = gt_candidate
            
            try:
                result = self.process_single(image_path, gt_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        return results
    
    def _save_results(self, result: Dict[str, Any]):
        """Save pipeline results to disk"""
        image_name = result['image_name']
        
        # Determine which subdirectory based on support usage
        if result['support_used']:
            subdir = 'img_img'  # Used image-image matching
        elif result['support_accepted']:
            subdir = 'img_text_accepted'  # High quality, added to bank
        else:
            subdir = 'img_text'  # Image-text only
        
        final_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(final_dir, exist_ok=True)
        
        # Save final mask
        if result['final_mask'] is not None:
            final_path = os.path.join(final_dir, f"{image_name}.png")
            cv2.imwrite(final_path, result['final_mask'])
            cv2.imwrite(os.path.join(self.output_dir, f"{image_name}.png"), result['final_mask'])
        
        # Optionally save intermediate results
        if self.debug:
            debug_dir = os.path.join(self.output_dir, 'debug', image_name)
            os.makedirs(debug_dir, exist_ok=True)
            
            if result['coarse_mask'] is not None:
                cv2.imwrite(
                    os.path.join(debug_dir, 'coarse.png'), 
                    result['coarse_mask']
                )
            
            if result['refined_mask'] is not None:
                cv2.imwrite(
                    os.path.join(debug_dir, 'refined.png'), 
                    result['refined_mask']
                )
    
    def get_support_bank_status(self) -> Dict[str, Any]:
        """Get support bank status for logging"""
        return {
            'bank_size': len(self.support_bank.samples),
            'score_threshold': self.support_bank.current_threshold,
            'n_accepted': len(self.support_bank.score_history),
            'is_ready': self.support_bank.is_ready(),
        }
        
# ============================================================================
# Evaluation Utilities
# ============================================================================

class Evaluator:
    """
    Evaluate segmentation results.
    Preprocessing and DSC computation match evaluation/eval.py:
    - threshold=200 for binarization
    - per-label DSC averaged across all non-zero labels
    """

    @staticmethod
    def _preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask to binary (0/255 uint8) matching evaluation/eval.py.
        Normalizes to 0-255 range, then thresholds at 200.
        """
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Dice coefficient matching evaluation/eval.py.
        Per-label DSC with edge case handling:
        - Both empty for a label -> DSC=1
        - GT empty but pred has the label -> DSC=0
        """
        gt_data = Evaluator._preprocess_mask(gt)
        seg_data = Evaluator._preprocess_mask(pred)

        gt_labels = np.unique(gt_data)[1:]   # skip 0 (background)
        seg_labels = np.unique(seg_data)[1:]
        labels = np.union1d(gt_labels, seg_labels)

        if len(labels) == 0:
            return 1.0  # Both masks empty

        DSC_arr = []
        for label in labels:
            if np.sum(gt_data == label) == 0 and np.sum(seg_data == label) == 0:
                DSC_i = 1.0
            elif np.sum(gt_data == label) == 0 and np.sum(seg_data == label) > 0:
                DSC_i = 0.0
            else:
                i_gt = (gt_data == label)
                i_seg = (seg_data == label)
                # Inline compute_dice_coefficient (same as evaluation/SurfaceDice.py)
                vol_sum = i_gt.sum() + i_seg.sum()
                vol_intersect = (i_gt & i_seg).sum()
                DSC_i = 2 * vol_intersect / vol_sum
            DSC_arr.append(DSC_i)

        return float(np.mean(DSC_arr))

    @staticmethod
    def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate IoU (Jaccard index)"""
        pred_binary = (Evaluator._preprocess_mask(pred) > 0).astype(np.float32)
        gt_binary = (Evaluator._preprocess_mask(gt) > 0).astype(np.float32)

        intersection = (pred_binary * gt_binary).sum()
        union = (pred_binary + gt_binary).clip(0, 1).sum()

        if union == 0:
            return 1.0
        return float(intersection / union)

    @staticmethod
    def precision(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate precision"""
        pred_binary = (Evaluator._preprocess_mask(pred) > 0).astype(np.float32)
        gt_binary = (Evaluator._preprocess_mask(gt) > 0).astype(np.float32)

        tp = (pred_binary * gt_binary).sum()
        fp = (pred_binary * (1 - gt_binary)).sum()

        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    @staticmethod
    def recall(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate recall (sensitivity)"""
        pred_binary = (Evaluator._preprocess_mask(pred) > 0).astype(np.float32)
        gt_binary = (Evaluator._preprocess_mask(gt) > 0).astype(np.float32)

        tp = (pred_binary * gt_binary).sum()
        fn = ((1 - pred_binary) * gt_binary).sum()

        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    @staticmethod
    def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate specificity"""
        pred_binary = (Evaluator._preprocess_mask(pred) > 0).astype(np.float32)
        gt_binary = (Evaluator._preprocess_mask(gt) > 0).astype(np.float32)

        tn = ((1 - pred_binary) * (1 - gt_binary)).sum()
        fp = (pred_binary * (1 - gt_binary)).sum()

        if tn + fp == 0:
            return 0.0
        return float(tn / (tn + fp))

    @classmethod
    def compute_all_metrics(cls, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Compute all segmentation metrics."""
        return {
            'dice': cls.dice_score(pred, gt),
            'iou': cls.iou_score(pred, gt),
            'precision': cls.precision(pred, gt),
            'recall': cls.recall(pred, gt),
            'specificity': cls.specificity(pred, gt)
        }
    
    @classmethod
    def evaluate_all(cls, results: List[Dict], gt_dir: str) -> Dict[str, Any]:
        """
        Evaluate all results against ground truth.
        
        Args:
            results: List of result dictionaries
            gt_dir: Directory containing GT masks
            
        Returns:
            Dictionary with evaluation summary
        """
        metrics = {
            'img_text': defaultdict(list),
            'img_img': defaultdict(list),
            'all': defaultdict(list)
        }
        
        # Store per-image metrics for detailed analysis
        per_image_metrics = []
        
        for result in results:
            image_name = result['image_name']
            
            # Try multiple possible GT file extensions
            gt_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidate = os.path.join(gt_dir, f"{image_name}{ext}")
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            
            if gt_path is None:
                print(f"Warning: No GT found for {image_name}")
                continue
            
            try:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    print(f"Warning: Cannot load GT: {gt_path}")
                    continue
            except Exception as e:
                print(f"Warning: Error loading GT {gt_path}: {e}")
                continue
            
            # Resize GT to match prediction size if needed
            if 'image' in result and result['image'] is not None:
                gt = ImageProcessor.resize_to_match(gt, result['image'])
            
            final_mask = result['final_mask']
            refined_mask = result['refined_mask']
            
            # Compute metrics for final mask
            if final_mask is not None:
                final_metrics = cls.compute_all_metrics(final_mask, gt)
                
                # Determine which group this result belongs to
                key = 'img_img' if result.get('support_used', False) else 'img_text'
                
                for metric_name, value in final_metrics.items():
                    metrics[key][metric_name].append(value)
                    metrics['all'][metric_name].append(value)
                
                # Store per-image metrics
                per_image_metrics.append({
                    'image_name': image_name,
                    'score': result.get('score', 0.0),
                    'support_used': result.get('support_used', False),
                    **final_metrics
                })
        
        # Compute summary statistics
        summary = {}
        for key in ['img_text', 'img_img', 'all']:
            for metric_name in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                values = metrics[key].get(metric_name, [])
                if values:
                    summary[f'{key}_{metric_name}_mean'] = np.mean(values)
                    summary[f'{key}_{metric_name}_std'] = np.std(values)
                    summary[f'{key}_{metric_name}_median'] = np.median(values)
                    summary[f'{key}_{metric_name}_min'] = np.min(values)
                    summary[f'{key}_{metric_name}_max'] = np.max(values)
                    summary[f'{key}_count'] = len(values)
        
        summary['per_image'] = per_image_metrics
        
        return summary
    
    @classmethod
    def print_summary(cls, summary: Dict[str, Any]):
        """Print evaluation summary in a formatted way"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        # Overall metrics
        if 'all_dice_mean' in summary:
            print(f"\n{'Category':<15} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12}")
            print("-"*70)
            
            for category in ['all']:
                count_key = f'{category}_count'
                if count_key in summary and summary[count_key] > 0:
                    dice = summary.get(f'{category}_dice_mean', 0) * 100
                    dice_std = summary.get(f'{category}_dice_std', 0) * 100
                    iou = summary.get(f'{category}_iou_mean', 0) * 100
                    iou_std = summary.get(f'{category}_iou_std', 0) * 100
                    
                    print(f"{category:<15} {dice:>5.1f}±{dice_std:>4.1f}%  "
                          f"{iou:>5.1f}±{iou_std:>4.1f}%  "
                          )

        print("="*70 + "\n")
    
    @classmethod
    def save_summary(cls, summary: Dict[str, Any], output_path: str):
        """Save evaluation summary to file"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Write metrics table
            f.write(f"{'Category':<15} {'Dice':<15} {'IoU':<15} {'Precision':<15} {'Recall':<15}\n")
            f.write("-"*70 + "\n")
            
            for category in ['all']:
                count_key = f'{category}_count'
                if count_key in summary and summary[count_key] > 0:
                    dice = f"{summary.get(f'{category}_dice_mean', 0)*100:.2f}±{summary.get(f'{category}_dice_std', 0)*100:.2f}%"
                    iou = f"{summary.get(f'{category}_iou_mean', 0)*100:.2f}±{summary.get(f'{category}_iou_std', 0)*100:.2f}%"
                    prec = f"{summary.get(f'{category}_precision_mean', 0)*100:.2f}%"
                    recall = f"{summary.get(f'{category}_recall_mean', 0)*100:.2f}%"
                    count = summary[count_key]
                    
                    f.write(f"{category:<15} {dice:<15} {iou:<15} {prec:<15} {recall:<15} (n={count})\n")
            
            # Write per-image details
            # if 'per_image' in summary:
            #     f.write(f"\n{'='*70}\n")
            #     f.write(f"PER-IMAGE DETAILS\n")
            #     f.write(f"{'='*70}\n\n")
            #     f.write(f"{'Image':<25} {'Support':<10} {'Dice':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10}\n")
            #     f.write("-"*80 + "\n")
                
            #     for m in summary['per_image']:
            #         f.write(f"{m['image_name']:<25} "
            #                f"{'Yes' if m['support_used'] else 'No':<10} "
            #                f"{m['dice']*100:>6.2f}%  "
            #                f"{m['iou']*100:>6.2f}%  "
            #                f"{m['precision']*100:>6.2f}%  "
            #                f"{m['recall']*100:>6.2f}%\n")
            
            # # Write CSV of per-image metrics
            # csv_path = output_path.replace('.txt', '_per_image.csv')
            # if 'per_image' in summary:
            #     import pandas as pd
            #     df = pd.DataFrame(summary['per_image'])
            #     df.to_csv(csv_path, index=False)
            #     f.write(f"\nPer-image metrics saved to: {csv_path}\n")

# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming Zero-shot Lung Nodule Segmentation Pipeline"
    )
    
    # 输入输出
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--gt-dir', type=str, default=None,
                       help='Path to ground truth directory for evaluation')
    parser.add_argument('--mask-input', type=str, default=None,
                       help='预生成的 coarse mask 目录(与旧 at 管线一致:直接读 saliency 输出,'
                            '跳过 on-the-fly 显著性图生成)')
    
    # 模型路径
    parser.add_argument('--sam-checkpoint', type=str,
                       default='/home/psxll9/MedCLIP-SAM/checkpoint/sam_vit_h_4b8939.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--sam-model-type', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type')
    parser.add_argument('--clip-model-path', type=str,
                       default='/home/psxll9/v2/saliency_maps/model',
                       help='Path to finetuned CLIP model')
    
    # SAM 提示参数
    parser.add_argument('--prompts-type', type=str, default='points',
                       choices=['points', 'boxes', 'both'],
                       help='Type of SAM prompts')
    parser.add_argument('--num-points', type=int, default=30,
                       help='Number of positive points')
    parser.add_argument('--neg-num-points', type=int, default=10,
                       help='Number of negative points')
    parser.add_argument('--negative', action='store_true', default=False,
                       help='Whether to use negative points')
    parser.add_argument('--multimask', action='store_true', default=False,
                       help='Whether to use multimask output')
    parser.add_argument('--multicontour', action='store_true', default=False,
                       help='Whether to output multiple bounding boxes for each contour')
    parser.add_argument('--num-contours', type=int, default=2,
                       help='Number of top contours to keep after KMeans binarization')
    
    # 增强参数
    parser.add_argument('--num-augmentations', type=int, default=5,
                       help='Number of augmentations for TTA')
    parser.add_argument('--aug-flip', action='store_true', default=True,
                       help='Enable flip augmentation')
    parser.add_argument('--aug-rotate', action='store_true', default=True,
                       help='Enable rotation augmentation')
    parser.add_argument('--aug-scale', action='store_true', default=True,
                       help='Enable scale augmentation')
    
    # Support bank 参数
    parser.add_argument('--max-support-size', type=int, default=15,
                       help='Maximum support bank size')
    parser.add_argument('--score-percentile', type=float, default=80,
                       help='Percentile threshold for support selection')
    parser.add_argument('--min-warmup', type=int, default=10,
                       help='Minimum warmup samples before support selection')
    
    # 文本提示
    parser.add_argument('--text-prompts', type=str, nargs='+',
                       default=[
                           "lung",
                           "A medical chest X-ray showing findings suggestive of potential lung conditions.",
                       ],
                       help='Text prompts for CLIP scoring')
    
    # Prototype matching 参数
    parser.add_argument('--modality', type=str, default='lung',
                       choices=['opticdisc', 'lung'],
                       help='Modality type')
    parser.add_argument('--input-size', type=int, default=672,
                       help='Input resolution for prototype matching')
    parser.add_argument('--proto-grid', type=int, default=8,
                       help='Prototype pooling window')
    parser.add_argument('--lora', type=int, default=0,
                       help='LoRA parameter')
    parser.add_argument('--coarse-pred-only', type=str, default='True',
                       help='Coarse prediction only')
    
    # 通用参数
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-worker', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug output')
    # 显著性图参数
    parser.add_argument('--vlayer', type=int, default=7,
                       help='Vision layer for saliency')
    parser.add_argument('--vbeta', type=float, default=0.1,
                       help='Beta for saliency')
    parser.add_argument('--vvar', type=float, default=1.0,
                       help='Variance for saliency')
    
    # 超参数优化
    parser.add_argument('--hyper-opt', action='store_true', default=False,
                       help='Enable hyperparameter optimization')
    parser.add_argument('--val-path', type=str, default=None,
                       help='Path to validation set for hyperparameter optimization')
    parser.add_argument('--ensemble', action='store_true', default=False,
                       help='Use text ensemble for saliency maps')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    # 正常管线默认值(与旧 at 管线对齐):
    # - coarse on-the-fly 生成,不读预生成 saliency
    # - clipscore 用旧的两条文本
    # num-points/negative/neg-num-points 不在此覆盖——
    # parse_args 默认 30/False/10 即 MC 的旧配置,
    # Xray/Shenzhen 由各自 sh 显式传 20/--negative/20。
    if args.mask_input in (None, "", "none"):
        args.mask_input = None
    args.text_prompts = TEXT_PROMPTS_DEFAULT

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建配置
    config = PipelineConfig(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        clip_model_path=args.clip_model_path,
        text_prompts=args.text_prompts,
        prompts_type=args.prompts_type,
        num_points=args.num_points,
        neg_num_points=args.neg_num_points,
        use_negative=args.negative,
        multimask=args.multimask,
        multicontour=args.multicontour,
        num_contours=args.num_contours,
        num_augmentations=args.num_augmentations,
        aug_flip=args.aug_flip,
        aug_rotate=args.aug_rotate,
        aug_scale=args.aug_scale,
        max_support_size=args.max_support_size,
        score_threshold_percentile=args.score_percentile,
        min_warmup_samples=args.min_warmup,
        vlayer=args.vlayer,
        vbeta=args.vbeta,
        vvar=args.vvar,
        hyper_opt=args.hyper_opt,
        val_path=args.val_path,
        mask_input=args.mask_input,
        ensemble=args.ensemble,
        modality=args.modality,
        input_size=args.input_size,
        proto_grid=args.proto_grid,
        lora=args.lora,
        coarse_pred_only=args.coarse_pred_only,
        seed=args.seed,
        n_worker=args.n_worker,
        device=args.device,
        debug=args.debug,
    )

    print("=" * 60)
    print("Normal pipeline (on-the-fly image-text + bank image-image)")
    print(f"  mask-input : {config.mask_input} (None = on-the-fly coarse)")
    print(f"  hyper-opt  : {config.hyper_opt} (val-path: {config.val_path})")
    print(f"  text prompts: {config.text_prompts}")
    print(f"  num-points : {config.num_points}")
    print(f"  negative   : {config.use_negative} (neg={config.neg_num_points})")
    print("=" * 60)
    
    # 创建管道
    pipeline = StreamingLungPipeline(config)
    pipeline.set_output_dir(args.output)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 单张图片
        print(f"Processing single image: {args.input}")
        result = pipeline.process_single(args.input)
        
        if args.gt_dir and os.path.exists(args.gt_dir):
            gt_path = os.path.join(
                args.gt_dir, 
                os.path.basename(args.input).replace('.jpg', '.png').replace('.jpeg', '.png')
            )
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                dice = Evaluator.dice_score(result['final_mask'], gt)
                iou = Evaluator.iou_score(result['final_mask'], gt)
                print(f"\nResults for {result['image_name']}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Dice: {dice:.4f}")
                print(f"  IoU: {iou:.4f}")
                print(f"  Support used: {result['support_used']}")
    else:
        # 目录
        print(f"Processing directory: {args.input}")
        results = pipeline.process_directory(args.input, args.gt_dir)
        
        # 评估
        if args.gt_dir:
            print("\n" + "="*50)
            print("Evaluating results against ground truth...")
            print("="*50)
            
            summary = Evaluator.evaluate_all(results, args.gt_dir)
            
            # 打印汇总
            Evaluator.print_summary(summary)
            
            # 保存到文件
            # summary_path = os.path.join(args.output, 'evaluation_summary.txt')
            # Evaluator.save_summary(summary, summary_path)
            # print(f"Evaluation summary saved to: {summary_path}")
        
        # print(f"\nAll results saved to: {args.output}")
        # # 保存汇总
        # summary_path = os.path.join(args.output, 'summary.txt')
        # with open(summary_path, 'w') as f:
        #     f.write("Streaming Pipeline Summary\n")
        #     f.write("="*50 + "\n")
        #     f.write(f"Total images: {len(results)}\n")
            
        #     if args.gt_dir and 'all_dice_mean' in summary:
        #         f.write(f"\nImage-Text only:\n")
        #         f.write(f"  Dice: {summary.get('img_text_dice_mean', 0):.4f} ± {summary.get('img_text_dice_std', 0):.4f}\n")
        #         f.write(f"  IoU:  {summary.get('img_text_iou_mean', 0):.4f} ± {summary.get('img_text_iou_std', 0):.4f}\n")
        #         f.write(f"\nImage-Image (with support):\n")
        #         f.write(f"  Dice: {summary.get('img_img_dice_mean', 0):.4f} ± {summary.get('img_img_dice_std', 0):.4f}\n")
        #         f.write(f"  IoU:  {summary.get('img_img_iou_mean', 0):.4f} ± {summary.get('img_img_iou_std', 0):.4f}\n")
        
        # print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nOutput saved to: {args.output}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(**kwargs) -> StreamingLungPipeline:
    """
    Create a pipeline instance with custom configuration.
    
    Example:
        pipeline = create_pipeline(
            sam_checkpoint='path/to/sam.pth',
            max_support_size=15,
            debug=True
        )
        result = pipeline.process_single('image.png')
    """
    config = PipelineConfig(**kwargs)
    return StreamingLungPipeline(config)


def process_image(image_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Quick single image processing function.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        **kwargs: Additional pipeline configuration
    
    Returns:
        Processing result dictionary
    """
    pipeline = create_pipeline(**kwargs)
    pipeline.set_output_dir(output_dir)
    return pipeline.process_single(image_path)


def process_folder(input_dir: str, output_dir: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Quick folder processing function.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        **kwargs: Additional pipeline configuration
    
    Returns:
        List of processing result dictionaries
    """
    pipeline = create_pipeline(**kwargs)
    pipeline.set_output_dir(output_dir)
    return pipeline.process_directory(input_dir)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    main()