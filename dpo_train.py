import os
import torch
from typing import Optional, Dict
from dataclasses import dataclass, field, asdict
from datasets import DatasetDict,  load_dataset,  Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import DPOConfig
from trl import DPOConfig, DPOTrainer
from data import *
import random
import wandb
import json

def main(args):
    set_seed(args.seed)

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision = 'main',
        trust_remote_code=True,
        use_auth_token=True,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision = 'main',
        trust_remote_code=True,
        use_auth_token=True,
    )
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.padding_side:
        tokenizer.padding_side = "left"
      
    f = open(args.input_file, 'r')
    raw_datasets = Dataset.from_list(json.load(f))

    training_args = DPOConfig(
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        output_dir=args.output_dir,
        dataloader_drop_last=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reetrant),
        bf16=args.bf16,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        remove_unused_columns=False,
        run_name=args.run_name,
        report_to=args.report_to,
        logging_steps=args.logging_steps
    )
    
    
    trainer = DPOTrainer(
        policy_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=raw_datasets,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the policy model name"})
        ref_model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the reference model name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "bz"})
        learning_rate: Optional[float] = field(default=3e-7, metadata={"help": "learning rate"})
        beta: Optional[float] = field(default=0.01, metadata={"help": "beta"})
        lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "learning rate decay"})
        warmup_ratio: Optional[float] = field(default=0.01, metadata={"help": "warm up"})
        weight_decay: Optional[float] = field(default=0.00, metadata={"help": "weight decay"})
        seed: Optional[int] = field(default=30, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "gradient accumulation steps"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "None"})
        gradient_checkpointing_use_reetrant: Optional[bool] = field(default=False, metadata={"help": "None"})
        save_strategy: Optional[str] = field(default="no", metadata={"help": "no save during train"})
        output_dir: Optional[str] = field(default="./", metadata={"help": "directory"})
        input_file: Optional[str] = field(default="./", metadata={"help": "directory"})
        report_to: Optional[str] = field(default="none", metadata={"help": "wandb, none"})
        num_train_epochs: Optional[float] = field(default=1, metadata={"help": "training epoches"})
        truncation_mode: Optional[str] = field(default='keep_end', metadata={"help": "keep end"})
        logging_steps: Optional[str] = field(default=5, metadata={"help": "wandb, none"})
        run_name: Optional[str] = field(default="", metadata={"help": "run name"})

        


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    

    main(args)

    
