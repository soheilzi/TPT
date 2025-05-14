import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import argparse
import torch
import transformers
from transformers import Trainer, get_scheduler
from datasets import load_dataset
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import math
import os 
from utils.facal_loss_utils import FocalTrainer, ClipTrainer

def build_instruction_prompt(instruction: str):
    # Truncate the instruction to 1000 characters if it's longer
    truncated_instruction = instruction.strip()[:1000]
    # Build the instruction prompt
    return f""" You are an expert mathematician. 
            You are provided with a math problem.
            Your task is to solve the problem step-by-step, clearly showing all relevant calculations and reasoning.

            # PROBLEM:
            {truncated_instruction.strip()}
            
            Requirements:
            1. Provide a complete and correct solution in a markdown block.
            2. Explain each step of the solution in detail.
            3. Conclude with the final numerical answer on a new line in the format `#### [Answer]`, replacing `[Answer]` with the actual answer.

            # SOLUTION:
            """

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-2-2b-it")

@dataclass
class DataArguments:
    train_data_path: str = field(default="/data/math2b_2k.json")
    eval_data_path: str = field(default="/data/ev.json")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1600,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(default="gemma-tpt")
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    learning_rate: float = field(default=1e-06) #low lr for google models 
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    eval_accumulation_steps: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    eval_strategy: str = field(default="steps")
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=15)
    save_steps: int = field(default=100)  # Save less frequently if desired
    eval_steps: int = field(default=50)  # Evaluate less frequently if desired
    num_train_epochs: int = field(default=1)
    report_to: str = field(default="wandb")
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    load_best_model_at_end: bool = field(default=True)  # Load best model at the end
    save_total_limit: int = field(default=2)  # Keep only 2 checkpoints 
    loss_function: str = field(default="CE")  # Loss function to use

# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """Collects the state dict and dump to disk."""
#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict) # noqa

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """
    Safely saves the model state dict, tokenizer, and config to disk.
    Ensures CPU transfer to reduce memory consumption during save.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect model state dict
    state_dict = trainer.model.state_dict()
    
    if trainer.args.should_save:
        # Move all tensors to CPU to save memory
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        
        # Save the state dict manually
        torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save the model config and tokenizer
        trainer.model.config.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model, config, and tokenizer saved to: {output_dir}")

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    # Filter out examples that don't have a valid response
    valid_indices = [i for i, response in enumerate(examples['solution']) if response and response.strip()]
    # Filter sources and targets based on the valid indices
    sources = [
        build_instruction_prompt(examples['question'][i])
        for i in valid_indices
    ]
    targets = [
        f"{examples['solution'][i].strip()}{EOT_TOKEN}"
        for i in valid_indices
    ]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict



def train(args):

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Override with args from argparse if specified
    if args.train_data_path:
        data_args.train_data_path = args.train_data_path
    if args.eval_data_path:
        data_args.eval_data_path = args.eval_data_path
    if args.learning_rate:
        training_args.learning_rate = args.learning_rate
    if args.output_dir:
        training_args.output_dir = args.output_dir
    if args.loss_function:
        training_args.loss_function = args.loss_function

    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
        print(data_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # You can also set a custom pad token, like '[PAD]'


    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token:", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager' 
    )

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))

    train_datasets = load_dataset('json',data_files=data_args.train_data_path, split="train", cache_dir=training_args.cache_dir)

    eval_datasets = load_dataset( 'json', data_files=data_args.eval_data_path, split="train",  cache_dir=training_args.cache_dir)

    train_dataset = train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=train_datasets.column_names,
        load_from_cache_file=False, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    eval_dataset = eval_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=eval_datasets.column_names,
        load_from_cache_file=False, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    # print out a sample of your training data
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            # this is the decoded test and the matrix
            # print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    if training_args.loss_function == "CE":
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    elif training_args.loss_function == "FocalLoss2":
        trainer = FocalTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    elif training_args.loss_function == "Clip09":
        trainer = ClipTrainer(model=model, tokenizer=tokenizer, args=training_args, clip_val=0.9, **data_module)


    # Log metrics with wandb
    trainer.add_callback(transformers.integrations.WandbCallback())

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)



parser = argparse.ArgumentParser(description="Training script with custom arguments")

# Model and training parameters
parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2-2b-it", help="Path to pre-trained model or model name")
parser.add_argument("--loss_function", type=str, default="CE", help="loss functions", )

# Dataset paths
parser.add_argument("--train_data_path", type=str, default="data/math2b_2k.json", help="Path to the training dataset file (json)")
parser.add_argument("--eval_data_path", type=str, default="data/ev.json", help="Path to the evaluation dataset file (json")

# Training parameters
parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
parser.add_argument("--output_dir", type=str, default="gemma-tpt", help="Directory to save model and outputs")

# Parse arguments
args = parser.parse_args()

#TODO login to wandb and set a project name 
wandb.init(
    project="tpt-gemma",
    name= args.output_dir
)

IGNORE_INDEX = -100
EOT_TOKEN = ""


# Call train function
train(args)
