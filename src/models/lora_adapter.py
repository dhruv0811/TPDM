import math
import torch
import torch.nn as nn

class LoraAdapter(nn.Module):
    """Simple LoRA adapter for linear layers"""
    def __init__(self, original_module, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if isinstance(original_module, nn.Linear):
            self.lora_down = nn.Linear(original_module.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, original_module.out_features, bias=False)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
        else:
            raise ValueError(f"LoRA can only be applied to nn.Linear, not {type(original_module)}")
    
    def forward(self, x):
        # Original output
        original_output = self.original_module(x)
        
        # LoRA path
        lora_output = self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        
        # Combine outputs
        return original_output + lora_output