import torch
import torch.nn as nn
from typing import Dict, List, Optional, Type, Union, Tuple
from peft import LoraConfig, get_peft_model
from diffusers.models.attention_processor import Attention
from transformers import PreTrainedModel

class DiffusionLoraAdapter:
    """
    LoRA adapter for diffusion models.
    This class wraps a diffusion model and applies LoRA to specific components.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        model_type: str = "sd3",
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            model: The model to adapt (UNet component of diffusion model)
            lora_rank: Rank of LoRA adapters
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to
            model_type: Type of diffusion model ("sd3" or "sd1.5")
        """
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.model_type = model_type
        
        # Set default target modules based on model type if not provided
        if target_modules is None:
            if model_type == "sd3":
                self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            elif model_type == "sd1.5":
                self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.target_modules = target_modules
        
        self.peft_config = self._create_peft_config()
        self.adapted_model = self._adapt_model()
    
    def _create_peft_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",  # Default task type, will be ignored for our custom application
        )
    
    def _adapt_model(self) -> nn.Module:
        """Apply LoRA adaptation to the model."""
        # If model is already wrapped with PEFT, return as is
        if hasattr(self.model, "peft_config"):
            return self.model
        
        # Handle SD3 transformer model differently than SD1.5
        if self.model_type == "sd3":
            # For SD3, the main component is a transformer model
            # Need to adapt at the attention block level
            return self._adapt_sd3_model()
        else:
            # For SD1.5, we can use PEFT directly
            return get_peft_model(self.model, self.peft_config)
    
    def _adapt_sd3_model(self) -> nn.Module:
        """Apply LoRA specifically for SD3 transformer model."""
        # SD3 requires special handling due to its structure
        # Identify all attention modules and apply LoRA to them
        from peft.tuners.lora import LoraLayer
        
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules) and isinstance(module, nn.Linear):
                # Replace with LoRA equivalent
                parent_name = name.rsplit(".", 1)[0]
                parent = self.model.get_submodule(parent_name)
                child_name = name.rsplit(".", 1)[1]
                
                # Create LoRA layer
                lora_layer = LoraLayer(
                    module, 
                    self.lora_rank, 
                    self.lora_alpha, 
                    self.lora_dropout
                )
                
                # Replace original module with LoRA layer
                setattr(parent, child_name, lora_layer)
        
        # Add peft_config attribute to mark model as adapted
        self.model.peft_config = {"default": self.peft_config}
        return self.model
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters from the adapted model."""
        trainable_params = []
        for name, param in self.adapted_model.named_parameters():
            # Only include LoRA parameters
            if "lora_" in name and param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights."""
        adapter_state = {}
        for name, param in self.adapted_model.named_parameters():
            if "lora_" in name:
                adapter_state[name] = param.data.cpu()
        
        torch.save({
            "adapter_state": adapter_state,
            "config": {
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
                "model_type": self.model_type,
            }
        }, path)
    
    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights."""
        checkpoint = torch.load(path, map_location="cpu")
        adapter_state = checkpoint["adapter_state"]
        
        for name, param in self.adapted_model.named_parameters():
            if name in adapter_state:
                param.data.copy_(adapter_state[name])
        
        # Also update config if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            self.lora_rank = config.get("lora_rank", self.lora_rank)
            self.lora_alpha = config.get("lora_alpha", self.lora_alpha)
            self.lora_dropout = config.get("lora_dropout", self.lora_dropout)
            self.target_modules = config.get("target_modules", self.target_modules)
            self.model_type = config.get("model_type", self.model_type)