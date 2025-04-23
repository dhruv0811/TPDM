"""
LoRA-enabled SD3 Time Prediction Diffusion Model.
Extends the SD3PredictNextTimeStepModelRLOOWrapper to support LoRA training of the UNet component.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

from diffusers.utils import scale_lora_layers, unscale_lora_layers

from ..lora_adapter import DiffusionLoraAdapter
from .modeling_sd3_pnt import SD3PredictNextTimeStepModelRLOOWrapper, SD3PredictNextTimeStepModel

logger = logging.getLogger(__name__)


class SD3LoRAPredictNextTimeStepModel(SD3PredictNextTimeStepModel):
    """
    SD3 model with LoRA support for the UNet transformer component.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the SD3 model with LoRA support.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained SD3 model
            lora_rank: Rank of LoRA adapters
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            lora_target_modules: List of module names to apply LoRA to
            **kwargs: Additional arguments for SD3PredictNextTimeStepModel
        """
        # Filter out kwargs that the parent class doesn't accept
        parent_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['fsdp', 'max_inference_steps']}
        
        # Initialize the base model
        super().__init__(pretrained_model_name_or_path, torch_dtype=torch_dtype, **parent_kwargs)
        
        # Create LoRA adapter for transformer
        self.unet_lora_adapter = DiffusionLoraAdapter(
            model=self.transformer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            model_type="sd3"
        )
        
        # Replace transformer with LoRA-adapted version
        self.transformer = self.unet_lora_adapter.adapted_model
        
        logger.info(f"Initialized SD3 model with LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")


class SD3LoRAPredictNextTimeStepModelRLOOWrapper(SD3PredictNextTimeStepModelRLOOWrapper):
    """
    LoRA-enabled wrapper for the SD3 model with time prediction.
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
        Initialize the LoRA-enabled wrapper.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained SD3 model
            lora_rank: Rank of LoRA adapters
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            lora_target_modules: List of module names to apply LoRA to
            **kwargs: Additional arguments for SD3PredictNextTimeStepModelRLOOWrapper
        """
        # Instead of using the parent class initializer which creates a standard model,
        # we'll create our own initialization flow to use the LoRA-enabled model
        
        # Set basic attributes
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.agent_model = SD3LoRAPredictNextTimeStepModel(
            pretrained_model_name_or_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            **kwargs
        )
        
        # Set other attributes from kwargs
        self.relative = kwargs.get("relative", True)
        self.fsdp = kwargs.get("fsdp", [])
        self.max_inference_steps = kwargs.get("max_inference_steps", 28)
        
        # Make base model non-trainable
        self.agent_model.requires_grad_(False)
        
        # Make only the time predictor and LoRA parameters trainable
        self.agent_model.time_predictor.train()
        self.agent_model.time_predictor.requires_grad_(True)
        
        # Make sure LoRA parameters are trainable
        for name, param in self.agent_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        logger.info(f"Initialized SD3 LoRA+TPM model with LoRA rank={lora_rank}")
    
    def save_lora_weights(self, save_path: str):
        """
        Save only the LoRA weights to a file.
        
        Args:
            save_path: Path to save LoRA weights to
        """
        self.agent_model.unet_lora_adapter.save_adapter(save_path)
        logger.info(f"Saved LoRA weights to {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """
        Load LoRA weights from a file.
        
        Args:
            load_path: Path to load LoRA weights from
        """
        self.agent_model.unet_lora_adapter.load_adapter(load_path)
        logger.info(f"Loaded LoRA weights from {load_path}")
    
    def sample(self, inputs):
        """
        Generate model outputs using LoRA weights.
        
        Args:
            inputs: Dictionary of inputs for generation
            
        Returns:
            outputs: Dictionary of outputs after sampling
        """
        # Scale up LoRA layers for inference if needed
        lora_scale = inputs.get("lora_scale", 1.0)
        if lora_scale != 1.0:
            scale_lora_layers(self.agent_model, lora_scale)
            
        # Call the parent's sample method
        outputs = super().sample(inputs)
        
        # Unscale LoRA layers after inference if they were scaled
        if lora_scale != 1.0:
            unscale_lora_layers(self.agent_model, lora_scale)
            
        return outputs