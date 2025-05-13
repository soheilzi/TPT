import json
import random
import argparse
import os


def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data, file_path):
    """Saves a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def split_data(data, train_size):
    """
    Splits the data into train and eval sets.

    :param data: List of data entries.
    :param train_size: Number of entries to select for the training set.
    :return: (train_data, eval_data)
    """
    if train_size > len(data):
        train_size = len(data)

    # Shuffle the data and split
    random.shuffle(data)
    train_data = data[:train_size]
    eval_data = data[train_size:]

    # Limit eval set to 200 samples
    eval_data = eval_data[:200]
    return train_data, eval_data


def main(input_file, train_output, eval_output, train_size):
    """
    Main function to load, split, and save the dataset.

    :param input_file: Path to the input JSON file.
    :param train_output: Path to save the train JSON file.
    :param eval_output: Path to save the eval JSON file.
    :param train_size: Number of entries for the train set.
    """
    print(f"Loading data from {input_file}...")
    data = load_json(input_file)
    print(f"Total entries loaded: {len(data)}")

    print("Splitting data into train and eval sets...")
    train_data, eval_data = split_data(data, train_size)

    print(f"Train set size: {len(train_data)}")
    print(f"Eval set size (max 200): {len(eval_data)}")

    print(f"Saving train data to {train_output}...")
    save_json(train_data, train_output)

    print(f"Saving eval data to {eval_output}...")
    save_json(eval_data, eval_output)

    print("Data splitting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON dataset into train and eval sets.")
    parser.add_argument('--input', default="samples/math_train/correct_answers.json", required=True, help="Path to the input JSON file.")
    parser.add_argument('--train_output', type=str, default="data/next/train2k.json", help="Path to save the train JSON file.")
    parser.add_argument('--eval_output', type=str,  default="data/next/evnext.json", help="Path to save the eval JSON file.")
    parser.add_argument('--train_size', type=int, default=2000, help="Number of entries for the train set.")
    args = parser.parse_args()

    main(args.input, args.train_output, args.eval_output, args.train_size)
