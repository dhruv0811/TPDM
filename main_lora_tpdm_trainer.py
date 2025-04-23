"""
Joint LoRA and Time Prediction Diffusion Model Training Script.

This script trains both LoRA adapters for the base diffusion model and 
the Time Prediction Diffusion Model (TPDM) simultaneously.
"""

import logging
import os
import pathlib

import hydra
import torch
import transformers
from omegaconf import OmegaConf

from src.train.lora_rloo_trainer import JointLoRaRLOOTrainer
from src.train.callbacks import DiffusionWandbCallback
from src.train.config import ConfigPathArguments, CustomRLOOConfig

# Logging setup
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


def train(cfg, training_args):
    """
    Train a joint LoRA and Time Prediction model.
    
    Args:
        cfg: Configuration for model, dataset, etc.
        training_args: Training arguments
    """
    # Instantiate model based on configuration
    model = hydra.utils.instantiate(
        OmegaConf.load(cfg.model_config),
        init_alpha=training_args.init_alpha,
        init_beta=training_args.init_beta,
        relative=training_args.relative,
        prediction_type=training_args.prediction_type,
        fsdp=training_args.fsdp,
        max_inference_steps=training_args.max_inference_steps,
    )
    logger.info(f"Model loaded from {cfg.model_config}")
    
    # Load reward model
    reward_model = hydra.utils.instantiate(OmegaConf.load(cfg.reward_model_config)).eval()
    logger.info(f"Reward model loaded from {cfg.reward_model_config}")
    
    # Load dataset
    train_dataset = hydra.utils.instantiate(OmegaConf.load(cfg.train_dataset))
    logger.info(f"Train dataset loaded from {cfg.train_dataset}")
    
    # Load data collator
    data_collator = hydra.utils.instantiate(OmegaConf.load(cfg.data_collator))
    logger.info(f"Data collator loaded from {cfg.data_collator}")

    # Initialize the joint LoRA+TPDM trainer
    trainer = JointLoRaRLOOTrainer(
        config=training_args,
        policy=model,
        reward_model=reward_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        lora_learning_rate=training_args.lora_learning_rate,
        lora_weight_decay=training_args.lora_weight_decay,
        freeze_time_predictor=training_args.freeze_time_predictor,
    )

    # Add wandb callback if specified
    if "wandb" in training_args.report_to:
        wandb_callback = DiffusionWandbCallback(trainer=trainer)
        logger.info("wandb callback added")
        trainer.add_callback(wandb_callback)

    # Resume from checkpoint if available
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and args.resume_from_checkpoint is not None
    ) or os.path.isdir(args.resume_from_checkpoint):
        # Judge whether resume_from_checkpoint is a path
        if os.path.isdir(args.resume_from_checkpoint):
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save LoRA weights separately for easy loading
    lora_save_path = os.path.join(training_args.output_dir, "lora_weights.pt")
    
    if hasattr(model, "save_lora_weights"):
        model.save_lora_weights(lora_save_path)
        logger.info(f"Saved LoRA weights to {lora_save_path}")
    else:
        lora_state_dict = {
            k: v for k, v in model.state_dict().items() if "lora_" in k
        }
        torch.save(lora_state_dict, lora_save_path)
        logger.info(f"Saved LoRA weights to {lora_save_path}")


if __name__ == "__main__":
    # Parse arguments
    parser = transformers.HfArgumentParser((ConfigPathArguments, CustomRLOOConfig))
    
    # Add LoRA-specific arguments
    parser.add_argument("--lora_learning_rate", type=float, default=1e-4, 
                       help="Learning rate for LoRA parameters")
    parser.add_argument("--lora_weight_decay", type=float, default=0.0, 
                       help="Weight decay for LoRA parameters")
    parser.add_argument("--freeze_time_predictor", action="store_true", 
                       help="Whether to freeze the time predictor during training")
    
    cfg, args = parser.parse_args_into_dataclasses()

    # Set model-specific args if not provided
    if "sd3" in cfg.model_config:
        if not hasattr(args, "init_alpha") or args.init_alpha is None:
            args.init_alpha = 2.5
        if not hasattr(args, "init_beta") or args.init_beta is None:
            args.init_beta = 1.0
    elif "sd1_5" in cfg.model_config:
        if not hasattr(args, "init_alpha") or args.init_alpha is None:
            args.init_alpha = 1.5
        if not hasattr(args, "init_beta") or args.init_beta is None:
            args.init_beta = -0.7

    # Run training
    train(cfg, args)