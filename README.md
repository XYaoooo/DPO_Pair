# DPO_Pair
## Code function:
Train DPO model with: dpo_train.py  
Train sft model with: sft_train.py  
sample data with: sample_ultra.py  
compute reward with: reward_batch.py

## Data
[sample data and reward for llama3-base sft mode](https://huggingface.co/datasets/YaoYX/llama_sft_sample)   
[sample data and reward for mistral-base sft mode](https://huggingface.co/datasets/YaoYX/mistral_sft_sample)  
[sample data and reward for llama3-instruct mode](https://huggingface.co/datasets/YaoYX/llama_instruct_sample)  
[sample data and reward for mistral-instruct mode](https://huggingface.co/datasets/YaoYX/mistral_instruct_sample)  
## Some CKs

 ### ($\max, \min$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_-1000sigma)  

 ### ($\max, \mu-2\sigma$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_-2sigma)    


 ### ($\max, \mu-1\sigma$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_-sigma)    


 ### ($\max, \mu$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_0sigma)    

 ### ($\max, \mu+\sigma$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_1sigma)   

  ### ($\max, \mu+2\sigma$)  
 [mistral instruct](https://huggingface.co/YaoYX/Mistral_instruct_chosen_mu_1000sigma_rejected_mu_2sigma) 




 ### ($\max, \min$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_-1000sigma)  

 ### ($\max, \mu-2\sigma$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_-2sigma)    


 ### ($\max, \mu-1\sigma$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_-1sigma)    


 ### ($\max, \mu$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_0sigma)    

 ### ($\max, \mu+\sigma$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_1sigma)   

  ### ($\max, \mu+2\sigma$)  
 [llama instruct](https://huggingface.co/YaoYX/Llama_insruct_chosen_mu_1000sigma_rejected_mu_2sigma)

 ## Some CKs of our method
 [llama instruct 5 samples](https://huggingface.co/YaoYX/DPO_llama_instruct_5)  
 [llama instruct 20 samples](https://huggingface.co/YaoYX/DPO_llama_instruct_20)  
 [llama instruct 60 samples](https://huggingface.co/YaoYX/DPO_llama_instruct_60)  
 [llama instruct 100 samples](https://huggingface.co/YaoYX/DPO_llama_instruct_100)  
 [llama instruct 200 samples](https://huggingface.co/YaoYX/DPO_llama_instruct_200)  
