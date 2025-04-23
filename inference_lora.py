"""
Inference script for LoRA-enabled TPDM model.

This script loads a trained LoRA+TPDM model and generates images with it,
allowing control of the LoRA strength.
"""

import argparse
import os
import logging
import torch
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LoRA+TPDM Inference")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for LoRA weights")
    parser.add_argument("--output_dir", type=str, default="outputs/inference", help="Output directory")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare_with_base", action="store_true", help="Compare with base model (no LoRA)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration
    logger.info(f"Loading model from config: {args.model_config}")
    model = hydra.utils.instantiate(OmegaConf.load(args.model_config))
    
    # Load LoRA weights if they exist
    if os.path.exists(args.lora_path):
        logger.info(f"Loading LoRA weights from: {args.lora_path}")
        if hasattr(model, "load_lora_weights"):
            model.load_lora_weights(args.lora_path)
        else:
            # Fallback to manual loading
            lora_state_dict = torch.load(args.lora_path, map_location="cpu")
            if "adapter_state" in lora_state_dict:
                lora_state_dict = lora_state_dict["adapter_state"]
            model.load_state_dict(lora_state_dict, strict=False)
    else:
        logger.warning(f"LoRA weights not found at: {args.lora_path}")
    
    # Move model to GPU
    model = model.to(device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Generate images in batches
    prompts = [args.prompt] * args.batch_size
    negative_prompts = [args.negative_prompt] * args.batch_size if args.negative_prompt else None
    
    logger.info(f"Generating {args.num_images} images with prompt: '{args.prompt}'")
    
    # Generate images with LoRA
    for i in range(0, args.num_images, args.batch_size):
        batch_size = min(args.batch_size, args.num_images - i)
        if batch_size < args.batch_size:
            prompts = prompts[:batch_size]
            if negative_prompts:
                negative_prompts = negative_prompts[:batch_size]
        
        # Setup generation inputs
        inputs = {
            "prompt": prompts,
            "negative_prompt": negative_prompts,
            "lora_scale": args.lora_scale,
            "predict": True,
            "num_inference_steps": 28,
        }
        
        # Generate with LoRA
        with torch.no_grad():
            logger.info(f"Generating batch {i // args.batch_size + 1}/{(args.num_images + args.batch_size - 1) // args.batch_size}")
            outputs = model.sample(inputs)
            
            # Save images
            for j, image in enumerate(outputs["images"]):
                idx = i + j
                image_path = os.path.join(args.output_dir, f"lora_{idx}_scale_{args.lora_scale}.png")
                image.save(image_path)
                logger.info(f"Saved image to: {image_path}")
            
    # Generate with base model (no LoRA) for comparison if requested
    if args.compare_with_base:
        logger.info("Generating comparison images with base model (LoRA scale = 0.0)")
        
        # Set LoRA scale to 0.0 to disable LoRA
        inputs["lora_scale"] = 0.0
        
        with torch.no_grad():
            outputs = model.sample(inputs)
            
            # Save base model images
            for j, image in enumerate(outputs["images"]):
                image_path = os.path.join(args.output_dir, f"base_{j}.png")
                image.save(image_path)
                logger.info(f"Saved base model image to: {image_path}")
    
    logger.info("Image generation complete!")


if __name__ == "__main__":
    main()