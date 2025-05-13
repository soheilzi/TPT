import os
import re
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def create_dir(directory):
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def generate_code(messages, tokenizer, sampling_params, llm):
    """Generate responses from the model based on input messages."""
    prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]

def process_response(response):
    """Process the model's response, removing unnecessary formatting."""
    processed_response = re.sub(r"#### \[Answer\]", "####", response)
    processed_response = re.sub(r"(\*\*|####)?\s*\[Answer\](?:[:])?", "", processed_response)
    return processed_response

def generate_math_solutions(json_file, output_files, tokenizer, sampling_params, llm):
    messages = []
    nums = []
    questions = []

    # Load questions from the JSON file
    with open(json_file, 'r') as file:
        questions_data = json.load(file)

    for question in questions_data:
        num = question["idt"]  # Assuming 'id' is the unique identifier for each question
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

    # Generate responses using the llm and tokenizer
    for idx, output_file in enumerate(output_files):
        responses = generate_code(messages, tokenizer, sampling_params, llm)

        results = []
        for i in range(len(responses)):
            response = responses[i].strip()
            processed_response = process_response(response)

            result = {
                "qnum": nums[i],
                "question": questions[i],
                "solution": processed_response
            }
            results.append(result)

        with open(output_file, 'w') as out_file:
            json.dump(results, out_file, indent=4)
        print(f"Solutions saved to {output_file}")

def main(model_name, max_model_len, outputdir, num_samples, temperature):
    if "g" in model_name:
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1500, stop_token_ids=[tokenizer.eos_token_id])

    # Run the solution generation
    json_file = 'data/test500.json'

    fls = [f'{outputdir}/e{i}.json' for i in range(0, num_samples)]

    generate_math_solutions(json_file, fls, tokenizer, sampling_params, llm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math Evaluation Script")
    parser.add_argument('--model_name', type=str, default="google/gemma-2-2b-it", help="The name of the model to be used.")
    parser.add_argument('--max_model_len', type=int, default=1500, help="Maximum model length.")
    parser.add_argument('--outputdir', type=str, default="samples/math_eval/2b", help="output to store results.")
    parser.add_argument('--numsamples', type=int, default=10, help="Number of samples to process.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    create_dir(args.output_dir)
    main(args.model_name, args.max_model_len, args.outputdir, args.numsamples, args.temperature)
