
import json
import torch
from tqdm import tqdm
from typing import Dict, List
from typing import Optional, Dict
from dataclasses import dataclass, field, asdict
from datasets import DatasetDict,  load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser,  set_seed

from data import *


from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, batch_messages: List[List[Dict[str, str]]]) -> List[Dict[str, float]]:
        """
        batch_messages: A list of OpenAI chat messages, where each message is a list of dictionaries
        Returns: A list of dictionaries with scores between 0 and 1 for each batch
        """
        # Tokenize the batch
        
        batch_input = [self.tokenizer.apply_chat_template(
            messages, truncation=self.truncation, max_length=self.max_length, tokenize=False
        ) for messages in batch_messages]

        
        # Flatten tokenized results and convert to tensors
        tokenized_inputs = self.tokenizer(
            batch_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            score = outputs.logits.float().squeeze().tolist()
        
        
        return score
       



def main(args):
    set_seed(args.seed)
    
    rm = ArmoRMPipeline(args.model_name, trust_remote_code=True)
    
    instruction_text_map = {}
    tokenizer =  AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    for item in raw_datasets:
        ori_prompt = item["chosen"][:-1]
        maybe_insert_system_message(ori_prompt, tokenizer)
        format_prompt = tokenizer.apply_chat_template(
            ori_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        instruction_text_map[format_prompt.strip()] = ori_prompt
        
    res = []
    
    f = open(args.file + ".json")
    data = json.load(f)
    res = []
    for item in tqdm(data):
        temp_item = {}
        scores = []
        
        inst = item['instruction'].strip()
        prompt = instruction_text_map[inst]

        for resp_index in range(args.begin, args.end, args.bz):
            batch_prompt = []
            for resp in item["output"][resp_index : resp_index + args.bz]:
                batch_prompt.append(prompt + [{"role": "assistant", "content": resp}])
                
            score = rm(batch_prompt)
            scores = scores + score
        
        temp_item["scores"] = scores
        temp_item["output"] = item["output"][args.begin: args.end]
        temp_item['instruction']=item['instruction']
        
        res.append(temp_item)
        
    print("End **********************************")
    with open(f"/raid/longhorn/xiaoyao/llama_file/llama_instruct_sample_armorm_second_{args.begin}_{args.end}.json", 'w') as fout:
        json.dump(res, fout, indent = 6)

if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="RLHFlow/ArmoRM-Llama3-8B-v0.1", metadata={"help": "the policy model name"})
        tokenizer: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "tokenizer"})
        seed: Optional[int] = field(default=30, metadata={"help": "random seed"})
        bz: Optional[int] = field(default=10, metadata={"help": "bz"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        file: Optional[str] = field(default='/raid/longhorn/xiaoyao/llama_file/llama_instruct_sample_armorm_second', metadata={"help": "file"})
        begin: Optional[int] = field(default=0, metadata={"help": "index"})
        end: Optional[int] = field(default=20, metadata={"help": "index"})

    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    
    print(args)
    
    main(args)