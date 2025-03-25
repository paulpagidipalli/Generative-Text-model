import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Larger dataset for better training
dataset = [
    "Artificial intelligence is the future of technology",
    "Machine learning powers modern AI applications",
    "Deep learning improves computer vision and NLP",
    "Neural networks mimic the human brain",
    "Data science is transforming industries",
    "AI chatbots use natural language processing",
    "Reinforcement learning enhances AI decision making",
    "Computer vision enables facial recognition",
    "The evolution of AI is accelerating",
    "AI ethics is crucial for responsible development"
]

# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset)

# Save tokenizer
with open("../models/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer saved successfully with vocab size:", len(tokenizer.word_index) + 1)