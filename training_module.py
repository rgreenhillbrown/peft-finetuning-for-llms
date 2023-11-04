import os
import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          Trainer, TrainingArguments, 
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Config Grab
config_path = "./config.yaml"
config = load_config(config_path)

# Accessing configurations in code
LORA_CONFIG_PARAMS = config['LORA_CONFIG_PARAMS']
TRAINING_ARGS_PARAMS = config['TRAINING_ARGS_PARAMS']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)

def initialize_model(model_name, token):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map='auto',
        token=token  
    )
    for param in model.parameters():
        param.requires_grad = False  
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model

def configure_lora_adapters(model):
    lora_config = LoraConfig(**LORA_CONFIG_PARAMS)
    return get_peft_model(model, lora_config)

def preprocess_data(tokenizer):
    data = load_dataset("Abirate/english_quotes") # change this to whatever dataset you like
    data = data.map(lambda e: {"prediction": f"{e['quote']} ->: {e['tags']}"})
    return data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

def train_model(model, data, tokenizer):
    training_args = TrainingArguments(**TRAINING_ARGS_PARAMS)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=training_args,
        data_collator=data_collator
    )
    model.config.use_cache = False  
    trainer.train()

def share_adapters(model):
    model.push_to_hub(
        "Your huggingface repo goes here",
        use_auth_token=True,
        commit_message="initial basic training",
        private=True
    )

def save_adapters_locally(model, path):
    model.save_pretrained(path, "lora")

def run_training(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = initialize_model(model_name, token)
    model = configure_lora_adapters(model)
    data = preprocess_data(tokenizer)
    train_model(model, data, tokenizer)
    share_adapters(model)
    save_adapters_locally(model, "./local_adapters")
