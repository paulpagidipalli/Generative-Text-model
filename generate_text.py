from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load trained GPT-2 model and tokenizer
model_path = r"C:\Users\chari\Desktop\Codetech\GenerativeTextModel\models\gpt_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# ✅ Fix: Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token  

# Generate text function
def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,  # ✅ Fix: Add attention mask
            max_new_tokens=max_new_tokens,  # ✅ Increased from 50 to 100
            temperature=1.1,  # 🔥 More randomness
            top_k=40,  # 🔥 More focused word choices
            top_p=0.9,  # 🔥 Slightly tighter nucleus sampling
            repetition_penalty=1.3,  # 🔥 Reduces repetition
            do_sample=True,  # ✅ Enables sampling for diversity
            num_return_sequences=1,  # ✅ Generates a single output
            early_stopping=True,  # ✅ Ensures full sentence completion
            eos_token_id=tokenizer.eos_token_id  # ✅ Stops at proper sentence end
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate_text("The future of AI is"))
