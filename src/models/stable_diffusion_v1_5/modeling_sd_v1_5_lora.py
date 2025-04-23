"""
LoRA-enabled SD1.5 Time Prediction Diffusion Model.
Extends the SD15PredictNextTimeStepModel to support LoRA training of the UNet component.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

from diffusers.utils import scale_lora_layers, unscale_lora_layers

from ..lora_adapter import DiffusionLoraAdapter
from .modeling_sd_v1_5 import SD15PredictNextTimeStepModel

logger = logging.getLogger(__name__)


class SD15LoRAPredictNextTimeStepModel(SD15PredictNextTimeStepModel):
    """
    SD1.5 model with LoRA support for the UNet component.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the SD1.5 model with LoRA support.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained SD1.5 model
            lora_rank: Rank of LoRA adapters
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            lora_target_modules: List of module names to apply LoRA to
            **kwargs: Additional arguments for SD15PredictNextTimeStepModel
        """
        # Initialize the base model
        super().__init__(pretrained_model_name_or_path, **kwargs)
        
        # Default targets for SD1.5 if not provided
        if lora_target_modules is None:
            lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        
        # Create LoRA adapter for UNet
        self.unet_lora_adapter = DiffusionLoraAdapter(
            model=self.unet,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            model_type="sd1.5"
        )
        
        # Replace UNet with LoRA-adapted version
        self.unet = self.unet_lora_adapter.adapted_model
        
        # Set trainable parameters
        self.requires_grad_(False)
        self.time_predictor.requires_grad_(True)
        
        # Make LoRA parameters trainable
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        logger.info(f"Initialized SD1.5 model with LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")
    
    def save_lora_weights(self, save_path: str):
        """
        Save only the LoRA weights to a file.
        
        Args:
            save_path: Path to save LoRA weights to
        """
        self.unet_lora_adapter.save_adapter(save_path)
        logger.info(f"Saved LoRA weights to {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """
        Load LoRA weights from a file.
        
        Args:
            load_path: Path to load LoRA weights from
        """
        self.unet_lora_adapter.load_adapter(load_path)
        logger.info(f"Loaded LoRA weights from {load_path}")
    
    def forward(self, **kwargs):
        """
        Forward pass with LoRA scaling support.
        """
        # Scale up LoRA layers for inference if needed
        lora_scale = kwargs.pop("lora_scale", 1.0)
        if lora_scale != 1.0:
            scale_lora_layers(self.unet, lora_scale)
            
        # Call the parent's forward method
        outputs = super().forward(**kwargs)
        
        # Unscale LoRA layers after inference if they were scaled
        if lora_scale != 1.0:
            unscale_lora_layers(self.unet, lora_scale)
            
        return outputs
    
    def rloo_repeat(self, data, rloo_k=2):
        """
        Repeat data for RLOO training.
        
        Args:
            data: Dictionary of data to repeat
            rloo_k: Number of times to repeat
        
        Returns:
            data: Repeated data
        """
        return super().rloo_repeat(data, rloo_k)
    
    def sample(self, inputs):
        """
        Generate model outputs using LoRA weights.
        
        Args:
            inputs: Dictionary of inputs for generation
            
        Returns:
            outputs: Dictionary of outputs after sampling
        """
        # Scale up LoRA layers for inference if needed
        lora_scale = inputs.pop("lora_scale", 1.0)
        if lora_scale != 1.0:
            scale_lora_layers(self.unet, lora_scale)
            
        # Call the parent's sample method
        outputs = super().sample(**inputs)
        
        # Unscale LoRA layers after inference if they were scaled
        if lora_scale != 1.0:
            unscale_lora_layers(self.unet, lora_scale)
            
        return outputs