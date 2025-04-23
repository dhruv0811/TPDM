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
        torch_dtype=None,
        fsdp=None,
        max_inference_steps: int = 28,
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
            fsdp: FSDP configuration (used by wrapper only)
            max_inference_steps: Maximum number of inference steps
            **kwargs: Additional arguments for SD3PredictNextTimeStepModel
        """
        # Set basic attributes
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # We'll store the FSDP and max_inference_steps parameters but not pass them to the model
        self.fsdp = [] if fsdp is None else fsdp
        self.max_inference_steps = max_inference_steps
        
        # Create the agent model
        self.agent_model = SD3LoRAPredictNextTimeStepModel(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            **kwargs  # Pass other arguments to the model
        )
        
        # Set other attributes from kwargs
        self.relative = kwargs.get("relative", True)
        
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