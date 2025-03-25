from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ✅ Fix: Set padding token
tokenizer.pad_token = tokenizer.eos_token  

model = GPT2LMHeadModel.from_pretrained("gpt2")  # ✅ Use smaller model for speed

# ✅ Use a much smaller dataset for faster training
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)  # ✅ Reduced max length for speed
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ✅ Apply tokenization efficiently
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ Speed optimizations
training_args = TrainingArguments(
    output_dir="../models/gpt_model",
    eval_strategy="no",  # ✅ Disables evaluation
    save_strategy="epoch",
    logging_dir="../logs",
    per_device_train_batch_size=32,  # ✅ Increased batch size for efficiency
    num_train_epochs=1,  # ✅ Reduced epochs for faster training
    max_steps=200,  # ✅ Limit training steps for speed
    fp16=True,  # ✅ Mixed-precision training for faster execution on GPU
    warmup_steps=100,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Fine-tune model
trainer.train()

# Save model
model.save_pretrained("../models/gpt_model")
tokenizer.save_pretrained("../models/gpt_model")
print("GPT-2 model saved successfully.")