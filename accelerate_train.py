from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from datasets import load_dataset
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os
import json
import torch

VOCAB_SIZE = 52000
MAX_LEN = 512

train_batch_size = 60
eval_batch_size = 60
learning_rate = 1e-4
warmup_steps = 500
num_epochs = 40
gradient_accumulation_steps = 4
checkpointing_steps == "epoch"
output_dir = "RoBERTa"

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=514,
    num_attention_heads=12,  # 16 Large, 12 Medium
    num_hidden_layers=6,  # 24 Large, 6 Medium
    type_vocab_size=1,
    hidden_size=768,  # 1024 Large, 768 Medium
)
# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print("Num parameters: ", model.num_parameters())

tokenizer_folder = "tokenizer_S_b"

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)


# raw_dataset = load_dataset("spsither/tibetan_monolingual_S_cleaned_train_test", num_proc=23)


# Tokenize helper function
# def tokenize(batch):
# return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")

# Tokenize dataset
# tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

tokenized_dataset = load_dataset(
    "spsither/tibetan_monolingual_S_cleaned_train_test_tokenized", num_proc=23
)

# Initialize accelerator with bf16
accelerator = Accelerator()  # mixed_precision="bf16")
optimizer = AdamW(params=model.parameters(), lr=learning_rate)

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_dataset["test"],
    shuffle=False,
    batch_size=eval_batch_size,
    collate_fn=data_collator,
)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Instantiate learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=len(train_dataloader) * num_epochs,
)


# Training loop
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Evaluation step
    model.eval()
    eval_loss = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            eval_loss += outputs.loss.detach()

    eval_loss = eval_loss / len(eval_dataloader)
    print(f"Validation Loss: {eval_loss}")

    if checkpointing_steps == "epoch":
        output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        accelerator.save_state(output_dir)

# Save final model and results
if output_dir is not None:
    model.save_pretrained(output_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump({"final_eval_loss": float(eval_loss)}, f)