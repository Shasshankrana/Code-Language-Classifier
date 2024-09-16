from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import time


def predict_language(model_path, tokenizer_name, directory):
    """
    Predicts the language of files in a directory.

    Args:
        model_path: Path to the pre-trained model for language classification.
        tokenizer_name: Name of the tokenizer for the model.
        directory: Path to the directory containing text files.

    Prints the filename and predicted language for each file.
    Writes the throughput (files per second) to "metrics.txt".
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to('cpu')  # Move the model to CPU

    # Set the number of threads used by PyTorch to 4
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Track processing time and number of files
    start_time = time.time()
    processed_files = 0

    # Loop through files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):

            # Read file content
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='latin1') as file:
                        text = file.read()
                except Exception as e:
                    print(f"Failed to read file {filepath}: {e}")
                    continue

            # Tokenize and predict language
            inputs = tokenizer(text, max_length=512, truncation=True,
                               padding=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to('cpu')
            attention_mask = inputs['attention_mask'].to('cpu')

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                predicted_language = torch.argmax(outputs.logits).item()

            # Print prediction
            print(
                f"Filename: {filename}, Predicted Language: {predicted_language}")

            processed_files += 1

    # Calculate and write throughput metrics
    end_time = time.time()
    total_time = end_time - start_time
    throughput = processed_files / total_time

    print(f"Throughput: {throughput:.2f} files/second\n")

    with open("metrics.txt", "a") as f:
        f.write(f"Throughput: {throughput:.2f} files/second\n")
        f.write(f"Processed Files: {processed_files}\n")


# Example usage (replace with your model and directory paths)
model_path = './results/final_model'
tokenizer_name = 'prajjwal1/bert-mini'
directory = './dataset/'

predict_language(model_path, tokenizer_name, directory)
