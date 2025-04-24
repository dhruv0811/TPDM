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
            orig_dtype = original_module.weight.dtype
            self.lora_down = nn.Linear(original_module.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, original_module.out_features, bias=False)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            
            # Convert to original dtype
            self.lora_down.weight.data = self.lora_down.weight.data.to(orig_dtype)
            self.lora_up.weight.data = self.lora_up.weight.data.to(orig_dtype)
        else:
            raise ValueError(f"LoRA can only be applied to nn.Linear, not {type(original_module)}")
    
    def forward(self, x):
        # Original output
        original_output = self.original_module(x)
        
        # Ensure x is the right dtype for LoRA
        orig_dtype = x.dtype
        
        # LoRA path (keeping the same dtype throughout)
        lora_output = self.lora_down(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_up(lora_output)
        lora_output = lora_output * self.scaling
        
        # Combine outputs (both should now be the same dtype)
        return original_output + lora_output