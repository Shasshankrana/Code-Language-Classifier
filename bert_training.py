import transformers
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Define the dataset directory
dataset_dir = './FileTypeData/'

# Function to load files and create dataset
def load_files_and_create_dataset(dataset_dir):
    texts = []
    labels = []

    # Get all subdirectories (languages)
    language_directories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Dynamically determine the number of classes
    n_classes = len(language_directories)

    # Assign a label to each directory
    label_mapping = {language: idx for idx, language in enumerate(language_directories)}

    def read_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin1') as file:
                    return file.read()
            except Exception as e:
                print(f"Failed to read file {filepath}: {e}")
                return None

    for language in language_directories:
        language_dir = os.path.join(dataset_dir, language)
        for filename in os.listdir(language_dir):
            filepath = os.path.join(language_dir, filename)
            if os.path.isfile(filepath):
                content = read_file(filepath)
                if content is not None:
                    texts.append(content)
                    labels.append(label_mapping[language])

    # Create a dictionary with the data
    data = {
        "text": texts,
        "label": labels
    }

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)

    return dataset, n_classes

# Load your custom dataset and get number of classes
dataset, n_classes = load_files_and_create_dataset(dataset_dir)

# Load the tokenizer and model with dynamic n_classes
tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
model = transformers.AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-mini', num_labels=n_classes)

train_val_test_split = dataset.train_test_split(test_size=0.2)
train_val_split = train_val_test_split['train'].train_test_split(test_size=0.125)

train_dataset = train_val_split['train']  # 70% of the data
val_dataset = train_val_split['test']    # 10% of the data
test_dataset = train_val_test_split['test'] # 20% of the data

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision.mean(),  # Weighted average precision
        "recall": recall.mean(),      # Weighted average recall
        "f1": f1.mean()               # Weighted average F1 score
    }

# Define training arguments
training_args = transformers.TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.05,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
eval_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)

# Extract predictions and labels for detailed metrics
def extract_predictions_and_labels(trainer, test_dataset):
    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=-1)
    return predicted_labels, labels

predicted_labels, true_labels = extract_predictions_and_labels(trainer, tokenized_test_dataset)

language_directories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
label_mapping = {language: idx for idx, language in enumerate(language_directories)}
class_names = [language for language, idx in sorted(label_mapping.items(), key=lambda item: item[1])]


result = confusion_matrix(true_labels, predicted_labels)
print(result)

report = classification_report(true_labels, predicted_labels, target_names=class_names)
print(report)


# Define the width for each column
col_width = 12

# Write the results to a file with properly spaced class names in the confusion matrix
with open("metrics.txt", "w") as resultfile:
    resultfile.write("Predicted on " + str(len(tokenized_test_dataset)) + " files. Results are as follows:\n\n")

    # Write the confusion matrix with class names
    resultfile.write("Confusion Matrix:\n")

    # Write the header with class names, spaced properly
    header = "".join(f"{class_name:<{col_width}}" for class_name in [""] + class_names) + "\n"
    resultfile.write(header)

    # Write each row of the confusion matrix with the corresponding class name, spaced properly
    for class_name, row in zip(class_names, result):
        row_string = "".join(f"{value:<{col_width}}" for value in row)
        resultfile.write(f"{class_name:<{col_width}}{row_string}\n")

    resultfile.write("\nClassification Report\n")
    resultfile.write(report)


# Save the trained model
model.save_pretrained('./results/final_model')

print("Metrics and model saved successfully.")
