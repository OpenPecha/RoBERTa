from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from datasets import load_dataset
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments

VOCAB_SIZE = 52000
MAX_LEN = 512

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


dataset = load_dataset("spsither/tibetan_monolingual_S_cleaned_train_test", num_proc=23)


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=512):
        self.df = pd.DataFrame(dataset["text"])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        inputs = self.tokenizer.encode_plus(
            self.df.iloc[i, 0],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
        }


eval_dataset = CustomDataset(dataset["test"], tokenizer)

train_dataset = CustomDataset(dataset["train"], tokenizer)


# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# Define the training arguments
training_args = TrainingArguments(
    output_dir="RoBERTa",
    overwrite_output_dir=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=40,
    learning_rate=1e-4,  # default: 0.001
    warmup_steps=500,
    weight_decay=0.01,
    per_device_train_batch_size=60,  # 1-32 # 60-1gpu TTF # 30-1gpu FFF # 4-4gpus TTT
    per_device_eval_batch_size=60,  # can be larger than per_device_train_batch_size, no need for grad
    gradient_checkpointing=True,  # default False, try False
    fp16=True,  # default False, try False
    group_by_length=True,  # default False, try False # takes time
    gradient_accumulation_steps=1,  # default 1
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=40,
    report_to=["wandb"],
)
# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    # prediction_loss_only=True,
)

# Train the model
trainer.train()
