#!/bin/bash

# Helper function to check the success of a command
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: Previous command failed. Exiting..."
        exit 1
    fi
}

echo "Starting Sequential Jobs..."

# First Job Placeholder (if applicable)
echo "Running ft jobs..."
# python3.10 first_job_script.py --args
# check_success


  # python3.10 ml/newsft/sftrev1.py --model_name_or_path  "meta-llama/Llama-3.2-1B-Instruct" --train_data_path ml/1Data/lam/cc/lamcc_2b1k.json --eval_data_path ml/1Data/lam/cc/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-cc-lam2b1k
  # check_success


  # python3.10 ml/newsft/sftrev1.py --model_name_or_path  "meta-llama/Llama-3.2-1B-Instruct" --train_data_path ml/1Data/lam/cc/lamcc_2b1k.json --eval_data_path ml/1Data/lam/cc/ev.json --learning_rate 2e-6 --output_dir ./imp/gemma-cc-lam2b1kk
  # check_success


  # python3.10 ml/newsft/sftrev1.py --model_name_or_path "google/gemma-2-2b-it" --train_data_path ml/1Data/other/cc/1k_2b_noprune.json --eval_data_path   ml/1Data/other/cc/ev/ev_5k_2b_noprune.json --learning_rate 1e-6 --output_dir ./imp/gemma-1k2bnoprunecc
  # check_success
  # python3.10 ml/newsft/sftrev1.py --model_name_or_path "google/gemma-2-2b-it" --train_data_path ml/1Data/other/cc/5k_2b_noprune.json  --eval_data_path   ml/1Data/other/cc/ev/ev_5k_2b_noprune.json  --learning_rate 1e-6 --output_dir ./imp/gemma-5k2bnoprunecc
  # check_success
  
  # python3.10 ml/newsft/sftrev1.py --model_name_or_path "google/gemma-2-2b-it" --train_data_path ml/1Data/other/cc/1k_9b_noprune.json  --eval_data_path   ml/1Data/other/cc/ev/ev_1k_9b_noprune.json  --learning_rate 1e-6 --output_dir ./imp/gemma-1k9bnoprunecc
  # check_success
  # python3.10 ml/newsft/sftrev1.py --model_name_or_path "google/gemma-2-2b-it" --train_data_path ml/1Data/other/cc/5k_9b_noprune.json  --eval_data_path   ml/1Data/other/cc/ev/ev_5k_9b_noprune.json --learning_rate 1e-6 --output_dir ./imp/gemma-5k9bnoprunecc
  # check_success


  # python3.10 ml/newsft/sftrev1.py --model_name_or_path ./imp/gemma-cc-2b1krec2 --train_data_path ml/1Data/2b/cc/rec3/rec3_2b_cc_1k.json --eval_data_path ml/1Data/2b/cc/rec3/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-cc-2b1krec3
  # check_success


  # python3.10 ml/newsft/sftrev1.py --model_name_or_path ./imp/gemma-cc-2b1krec2 --train_data_path ml/1Data/2b/cc/rec3/rec3_2b_cc_1kk.json --eval_data_path ml/1Data/2b/cc/rec3/evv.json --learning_rate 1e-6 --output_dir ./imp/gemma-cc-2b1krec32
  # check_success

  # python3.10 ml/newsft/sftrev1.py --model_name_or_path ./imp/gemma-leet-rec2-2b1k --train_data_path ml/1Data/2b/leet/rec3/rec3_2b_leet_1k.json --eval_data_path  ml/1Data/2b/leet/rec3/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-leet-rec3-2b1k
  # check_success

  python3.10 ml/newsft/sft_math.py  --model_name_or_path  "google/gemma-2-2b-it" --train_data_path ml/1Data/2b/math/math2b_1k.json --eval_data_path  ml/1Data/2b/math/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-mth-2b1k
  check_success


  python3.10 ml/newsft/sft_math.py  --model_name_or_path  "google/gemma-2-2b-it" --train_data_path ml/1Data/9b/math/math9b_1k.json --eval_data_path  ml/1Data/9b/math/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-mth-9b1k
  check_success



  # python3.10 ml/newsft/sftrev1.py --model_name_or_path ./imp/gemma-cc-2b1krec1 --train_data_path ml/1Data/2b/cc/rec2/rec2_2b_cc_1k.json --eval_data_path ml/1Data/2b/cc/rec2/ev.json --learning_rate 1e-6 --output_dir ./imp/gemma-cc-2b1krec2
  # check_success
  
  # python3.10 ml/newsft/sft_math.py  --model_name_or_path "google/gemma-2-2b-it" --train_data_path ml/1Data/other/math/6k_math_noprune.json --eval_data_path  ml/1Data/other/math/ev/ev_6k_math_noprune.json --learning_rate 1e-6 --output_dir ./imp/gemma-6k2bnoprunemath
  # check_success

# ------------------ Second Job ------------------
echo "Starting CodeC jobs..."


  # python3.10 ml/newsft/dp.py 

  # python3.10 CodeC/gen.py --model_name ./imp/gemma-cc-2b1krec3 --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/rec32b1k"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/rec32b1k --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name ./imp/gemma-cc-2b1krec32 --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/rec3"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/rec3 --samples 50
  # check_success




  # python3.10 CodeC/gen.py --model_name "meta-llama/Llama-3.2-1B-Instruct"  --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/lambase"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/lambase --samples 50
  # check_success


  # python3.10 CodeC/gen.py --model_name ./imp/gemma-cc-lam2b1k/checkpoint-125 --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/lam2b1k"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/lam2b1k --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name ./imp/gemma-cc-lam2b1kk/checkpoint-125 --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/lam2b1k"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/lam2b1k --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name ./imp/gemma-cc-2b1krec2 --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/rec22b1k"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/rec22b1k --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name  ./imp/gemma-1k2bnoprunecc --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/1k-2b-noprune"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/1k-2b-noprune --samples 50
  # check_success


  # python3.10 CodeC/gen.py --model_name  ./imp/gemma-5k2bnoprunecc --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/5k-2b-noprune"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/5k-2b-noprune --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name  ./imp/gemma-1k9bnoprunecc --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/1k-9b-noprune"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/1k-9b-noprune --samples 50
  # check_success

  # python3.10 CodeC/gen.py --model_name  ./imp/gemma-5k9bnoprunecc --max_model_len 1500  --outputdir "CodeC/jcccosine/0.7/5k-9b-noprune"  --numsamples 50  --temp 0.7
  # check_success

  # python3.10 CodeC/timenew.py --dirs CodeC/jcccosine/0.7/5k-9b-noprune --samples 50
  # check_success



# Add remaining CodeC commands similarly...

echo "Finished CodeC jobs."
echo "Starting Math jobs..."

# ------------------ Math Jobs ------------------
  # python3.10 Math/gen.py --model_name "google/gemma-2-9b-it" --max_model_len 1500  --outputdir "Math/eval0.7/9b"  --numsamples 20  --temp 0.7
  # python3.10 Math/gen.py --model_name "./imp/gemma-mth-rec3-2b2k" --max_model_len 1500  --outputdir "Math/eval0.7/rec32b2k"  --numsamples 20  --temp 0.7
  # check_success
  python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/imp/gemma-mth-2b1k" --max_model_len 1500  --outputdir "Math/eval0.7/2b1k"  --numsamples 20 --temp  0.7
  check_success
  python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/imp/gemma-mth-9b1k" --max_model_len 1500  --outputdir "Math/eval0.7/9b1k"  --numsamples 20  --temp  0.7
  check_success
  # python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/mdnew/math/gemma-mth-9b2k" --max_model_len 1500  --outputdir "Math/eval0.7/9b2k"  --numsamples 20  --temp 0.7
  # check_success
  # python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/mdnew/math/gemma-mth-9b4k" --max_model_len 1500  --outputdir "Math/eval0.7/9b4k"  --numsamples 20  --temp  0.7
  # check_success
  # python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/mdnew/math/gemma-mth-9b6k" --max_model_len 1500  --outputdir "Math/eval0.7/9b6k"  --numsamples 20  --temp  0.7
  # check_success
  # python3.10 Math/gen.py --model_name "/home/ubuntu/CaiaMaiCode/mdnew/math/gemma-mth-2b2k" --max_model_len 1500  --outputdir "Math/eval0.7/2b"  --numsamples 20  --temp  0.7
  # check_success

  # python3.10 Math/gen.py --model_name  ./imp/gemma-mth-lam-2b2k/checkpoint-250 --max_model_len 1500  --outputdir "Math/eval0.7/lam2k"  --numsamples 20  --temp  0.7
  # check_success
  # python3.10 Math/gen.py --model_name  "meta-llama/Llama-3.2-1B-Instruct" --max_model_len 1500  --outputdir "Math/eval0.7/lambase"  --numsamples 20  --temp  0.7
  # check_success


# Add remaining Math commands similarly...

echo "Finished Math jobs."
echo "Starting Leet jobs..."

# ------------------ Leet Jobs ------------------

  # python3.10 Leet/justgen.py --model_name ./imp/gemma-leet-rec3-2b1k --max_model_len 1500  --outputdir "Leet/res0.7/rec32b1k"  --numsamples 20  --temp 0.7
  # check_success
  # python3.10 Leet/time.py --dirs Leet/res0.7/rec32b1k --samples 20


  # python3.10 syn/bt.py
  # python3.10 syn/bt2.py
  # python3.10 syn/bt3.py


  # python3.10 Leet/justgen.py --model_name ./imp/gemma-1k2bnopruneleet --max_model_len 1500  --outputdir "Leet/res0.7/1k2bnoprune"  --numsamples 20  --temp 0.7
  # check_success

  # python3.10 Leet/justgen.py --model_name ./imp/gemma-leet-r2-2b1k --max_model_len 1500  --outputdir "Leet/res0.7/rec22b1kk"  --numsamples 20  --temp 0.7
  # check_success


  # python3.10 Leet/time.py --dirs Leet/res0.7/rec22b1kk --samples 20
  # check_success


  # python3.10 Leet/justgen.py --model_name ./imp/gemma-1k9b --max_model_len 1500  --outputdir "Leet/res0.7/9b1k"  --numsamples 20  --temp 0.7
  # check_success


  # python3.10 Leet/time.py --dirs Leet/res0.7/9b1k --samples 20
  # check_success


  # python3.10 Leet/time.py --dirs Leet/res0.7/1k9bnoprune --samples 20
  # check_success


  # python3.10 /home/ubuntu/CaiaMaiCode/syn/bt2.py
  # check_success

  # python3.10 /home/ubuntu/CaiaMaiCode/syn/bt3.py
  # check_success




  # python3.10 Leet/justgen.py --model_name "/home/ubuntu/CaiaMaiCode/mdnew/leet/gemma-leet-9b1k" --max_model_len 1500  --outputdir "Leet/res0.7/9b1k"  --numsamples 20  --temp 0.7
  # check_success



# Add remaining Leet commands similarly...

echo "All jobs completed successfully!"
