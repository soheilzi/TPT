import os
import re
import time
import shutil
import pickle
import subprocess
import timeit
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json 
import torch

# helper
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# generating code vllm
def generate_code(messages, tokenizer, sampling_params, llm):
    prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]

def process_response(response):
    """Process the model's response, removing unnecessary formatting."""
    processed_response = re.sub(r"#### \[Answer\]", "####", response)
    processed_response = re.sub(r"(\*\*|####)?\s*\[Answer\](?:[:])?", "", processed_response)
    return processed_response

def genmath(json_file, outs, tokenizer, sampling_params, llm):
    messages = []
    nums = []
    questions = []

    # Load questions from the JSON file
    with open(json_file, 'r') as file:
        questions_data = json.load(file)

    for question in questions_data:
        num = question["id"]  # Assuming 'id' is the unique identifier for each question
        q = question["question"]
        questions.append(q)
        nums.append(num)
        # Construct the prompt message for each question
        mess = [
                {
                    "role": "user",
                    "content": f"""
                    You are an expert mathematician. 
                    You are provided with a math problem.
                    Your task is to solve the problem step-by-step, clearly showing all relevant calculations and reasoning.

                    # PROBLEM:
                    {q.strip()}
                    
                    Requirements:
                    1. Provide a complete and correct solution in a markdown block.
                    2. Explain each step of the solution in detail.
                    3. Conclude with the final numerical answer on a new line in the format `#### [Answer]`, replacing `[Answer]` with the actual answer.

                    # SOLUTION:
                    """
                }
            ]
        messages.append(mess)

    for file in outs:
        # Generate responses using the llm and tokenizer
        responses = generate_code(messages, tokenizer, sampling_params, llm)
        # Collect results and write them to the output JSON file
        results = []
        for i, num in enumerate(nums):

            response = responses[i].strip()

            processed_response = process_response(response)

            result = {
                "qnum": num,
                "question": questions[i],
                "solution": processed_response
            }
            results.append(result)

        with open(file, 'w') as out_file:
            json.dump(results, out_file, indent=4)

        print(f"Solutions saved to {file}")
    return messages, nums
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gemma-tpt', help="Model name or path")
    parser.add_argument("--max_model_len", type=int, default=1500, help="Max model length")
    parser.add_argument("--num_samples", type=int, default=2, help="num sols per question")
    parser.add_argument("--math", type=str, default='data/gsm8ktrain.json', help="Path to Math dataset")
    parser.add_argument("--output_dir", type=str, default='samples/math_train/ft', help="Output directory")
    args = parser.parse_args()

    if 'gemma' in args.model_name:
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
    
    create_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(model=args.model_name, tensor_parallel_size=1, max_model_len=args.max_model_len, trust_remote_code=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=1500, stop_token_ids=[tokenizer.eos_token_id])
    
    if args.math:
        dirs = [f'{args.output_dir}/e{i}.json' for i in range(10)]
        genmath(args.math, dirs, tokenizer, sampling_params, llm)
    
if __name__ == "__main__":
    main()

 