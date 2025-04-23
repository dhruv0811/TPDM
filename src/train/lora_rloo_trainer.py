"""
Joint LoRA and TPDM Trainer for diffusion models.
This trainer extends the CommonRLOOTrainer to handle joint training of the LoRA adapters for
the base diffusion model alongside the time prediction model.
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

from accelerate.utils import DistributedType
from trl.trainer.utils import disable_dropout_in_model

from .rloo_trainer import CommonRLOOTrainer

logger = logging.getLogger(__name__)


class JointLoRaRLOOTrainer(CommonRLOOTrainer):
    """
    Trainer for joint LoRA and TPDM training.
    
    This trainer extends CommonRLOOTrainer to allow joint training of:
    1. LoRA adapters for the base diffusion model (UNet)
    2. The Time Prediction Diffusion Model (TPDM)
    
    It introduces separate parameter groups and optimization strategies for each component.
    """
    
    def __init__(
        self,
        lora_learning_rate: float = 1e-4,
        lora_weight_decay: float = 0.0,
        freeze_time_predictor: bool = False,
        **kwargs
    ):
        """
        Initialize the joint trainer.
        
        Args:
            lora_learning_rate: Learning rate for LoRA parameters
            lora_weight_decay: Weight decay for LoRA parameters
            freeze_time_predictor: Whether to freeze the time predictor during training
            **kwargs: Additional arguments for CommonRLOOTrainer
        """
        self.lora_learning_rate = lora_learning_rate
        self.lora_weight_decay = lora_weight_decay
        self.freeze_time_predictor = freeze_time_predictor
        
        # Initialize parent trainer
        super().__init__(**kwargs)
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create separate optimizer parameter groups for LoRA and time predictor.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # Get LoRA parameters from the policy
        if hasattr(self.policy, "agent_model") and hasattr(self.policy.agent_model, "unet_lora_adapter"):
            lora_params = self.policy.agent_model.unet_lora_adapter.get_trainable_parameters()
        else:
            lora_params = []
            for name, param in self.policy.named_parameters():
                if "lora_" in name and param.requires_grad:
                    lora_params.append(param)
        
        # Get time predictor parameters
        if hasattr(self.policy, "agent_model") and hasattr(self.policy.agent_model, "time_predictor"):
            time_pred_params = list(self.policy.agent_model.time_predictor.parameters())
        else:
            # Fallback to looking for time_predictor in the model structure
            time_pred_params = []
            for name, module in self.policy.named_modules():
                if "time_predictor" in name:
                    time_pred_params.extend(list(module.parameters()))
        
        # Freeze time predictor if specified
        if self.freeze_time_predictor:
            for param in time_pred_params:
                param.requires_grad = False
            logger.info("Freezing time predictor parameters")
            
        # Create parameter groups with different learning rates
        optimizer_grouped_parameters = [
            {
                "params": lora_params,
                "lr": self.lora_learning_rate,
                "weight_decay": self.lora_weight_decay,
                "name": "lora_params"
            },
            {
                "params": time_pred_params if not self.freeze_time_predictor else [],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "name": "time_pred_params"
            }
        ]
        
        # Log parameter counts
        lora_param_count = sum(p.numel() for p in lora_params)
        time_pred_param_count = sum(p.numel() for p in time_pred_params)
        logger.info(f"Training {lora_param_count} LoRA parameters " +
                   f"and {time_pred_param_count if not self.freeze_time_predictor else 0} time predictor parameters")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = self.get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    def _log_parameter_info(self):
        """Log information about trainable parameters."""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.policy.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        lora_params = sum(p.numel() for n, p in self.policy.named_parameters() 
                          if "lora_" in n and p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        logger.info(f"LoRA parameters: {lora_params:,} ({lora_params/total_params:.2%})")
        
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """
        Train both the LoRA adapters and time predictor.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from or bool flag
        """
        # Log parameter information before training
        self._log_parameter_info()
        
        # The RLOO training process in the parent class already handles the joint training
        return super().train(resume_from_checkpoint=resume_from_checkpoint)