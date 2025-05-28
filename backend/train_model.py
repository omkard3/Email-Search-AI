import pandas as pd
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch

# Constants
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # MUCH smaller model
MODEL_DIR = "backend/summarizer_model"
CSV_PATH = "data/processed_email_summarization_dataset.csv"
MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 96

def load_dataset(path):
    # Limit to 500 rows for fast execution
    df = pd.read_csv(path)[["emails_text", "summary"]].dropna().iloc[:500]
    return Dataset.from_pandas(df)

def tokenize_function(example, tokenizer):
    inputs = tokenizer(
        example["emails_text"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            example["summary"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True
        )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }

def train():
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    raw_dataset = load_dataset(CSV_PATH)
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["emails_text", "summary"]
    )
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir="backend/model_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,  # Faster
        logging_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),  # Use mixed precision
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train()
