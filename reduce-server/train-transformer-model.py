#!/usr/bin/env python3

import subprocess
subprocess.run("pip install -U 'numpy<1.24.0' accelerate transformers 'dill<0.3.5' datasets", shell=True)

import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print('import torch')
import torch

print('import transformers')
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import Trainer, TrainingArguments

print('init tokenizer')
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

print('init model')
model = BloomForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    #torch_dtype=torch.float16, -> Attempting to unscale FP16 gradients.
)

#model = model.to(torch.float16) -> Attempting to unscale FP16 gradients.

print('load dataset')
from datasets import load_dataset
dataset = load_dataset("pile-of-law/pile-of-law",'r_legaladvice')

def tokenize_function(examples):
    return tokenizer(examples["text"])

import os

print('preprocess dataset')
tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True, 
                                num_proc=os.cpu_count(),
                                remove_columns=["text","created_timestamp","downloaded_timestamp","url"],
)

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
 k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=os.cpu_count(),
)

print('creating trainer')
training_args = TrainingArguments(
    f"bloom560m-finetuned-pileoflaw_reddit",
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    optim="adafactor",
    logging_steps=40,
    save_strategy='epoch',
    weight_decay=0.1,
    learning_rate=5e-6,
    evaluation_strategy='steps',
    eval_steps=400,
    per_device_eval_batch_size=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()
