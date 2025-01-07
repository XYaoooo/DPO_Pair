import gc
import torch
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import datasets
import json
from datasets import load_dataset
from data import *

def sample_batch(instructions, model_tag, seed, n):
    
    tokenizer = AutoTokenizer.from_pretrained(model_tag, trust_remote_code=True)
        
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LLM(model=model_tag, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, swap_space=8)

    sampling_params = SamplingParams(n=n,
                                    temperature=0.8,
                                    max_tokens=4096,
                                    stop=[tokenizer.eos_token, "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"], seed=seed)
    
    outputs = model.generate(instructions, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    return_outputs = []
    
    for output in outputs:
        
        temp_output = { 
                        "instruction": output.prompt,
                        "output": [o.text for o in output.outputs],
        }
        
        return_outputs.append(temp_output)
    
    return return_outputs



def main(model, begin, end, seed, n_prompt, n):
    
    raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    
    column_names = list(raw_datasets.features)
    
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "generation",
            "auto_insert_empty_system_msg": True,
        },
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

   
    prompts = []
    
    if n_prompt > 0:
        raw_datasets = raw_datasets.shuffle(seed=42).select(range(1000))
    
    for example in raw_datasets:
        prompts.append(example["text"])
       
    return_outputs = sample_batch(prompts[begin:end], model, seed, n)
    
    with open(f'./llama_file/{begin}—{end}-{n}-{seed}.json', 'w') as fout:
        json.dump(return_outputs, fout, indent = 6)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # princeton-nlp/Llama-3-Base-8B-SFT    meta-llama/Meta-Llama-3-8B-Instruct
    parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct", help='max token length', type=str)
    parser.add_argument('--begin', action='store', default=0, help='begin', type=int)
    parser.add_argument('--end', action='store', default=8000, help='end', type=int)
    parser.add_argument('--seed', action='store', default=42, help='seed', type=int)
    parser.add_argument('--n_prompt', action='store', default=-1, help='seed', type=int)
    parser.add_argument('--n', action='store', default=25, help='samples', type=int)
 
 
    args = parser.parse_args()
    args = vars(args)

    main(**args)