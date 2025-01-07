# Fine-Tune Llama2-7b on SE paired dataset
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
from trl import SFTConfig, SFTTrainer
from data import *
import random
import wandb
import json

DEFAULT_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
MISTRAL_CHAT_TEMPLATE = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"


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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.padding_side:
        tokenizer.padding_side = "left"
   
    if "mistral" in args.model_name.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    else:  
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    column_names = list(raw_datasets.features)
    
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": True,
        },
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    
    training_args = SFTConfig(
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        output_dir=args.output_dir,
        dataloader_drop_last=False,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reetrant),
        bf16=args.bf16,
        max_seq_length=args.max_length,
        remove_unused_columns=False,
        run_name=args.run_name,
        report_to=args.report_to,
        logging_steps=args.logging_steps
    )
    
    trainer = SFTTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=raw_datasets,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.2", metadata={"help": "the policy model name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        batch_size: Optional[int] = field(default=4, metadata={"help": "bz"})
        learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "learning rate"})
        lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "learning rate decay"})
        warmup_ratio: Optional[float] = field(default=0.01, metadata={"help": "warm up"})
        weight_decay: Optional[float] = field(default=0.00, metadata={"help": "weight decay"})
        seed: Optional[int] = field(default=30, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "gradient accumulation steps"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "None"})
        gradient_checkpointing_use_reetrant: Optional[bool] = field(default=False, metadata={"help": "None"})
        save_strategy: Optional[str] = field(default="no", metadata={"help": "no save during train"})
        output_dir: Optional[str] = field(default="./", metadata={"help": "directory"})
        report_to: Optional[str] = field(default="none", metadata={"help": "wandb, none"})
        num_train_epochs: Optional[float] = field(default=1, metadata={"help": "training epoches"})
        logging_steps: Optional[str] = field(default=5, metadata={"help": "wandb, none"})
        run_name: Optional[str] = field(default="", metadata={"help": "run name"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    

    main(args)
