import tokenizer
import random
import numpy as np
import os

# PART A

# sample 10 documents from tinystories
train_path = 'data/TinyStoriesV2-GPT4-train.txt'
valid_path = 'data/TinyStoriesV2-GPT4-valid.txt'

with open(train_path, "r", encoding="utf-8") as f:
    train_text = f.read()
with open(valid_path, "r", encoding="utf-8") as f:
    valid_text = f.read()

documents = valid_text.split('<|endoftext|>')
ten_documents = random.sample(documents, 10)

# encode samples into IDs
tok = tokenizer.Tokenizer.from_files('data/pickle/tinystories_vocab', 'data/pickle/tinystories_merges', ['<|endoftext|>'])
total_num_bytes = 0
total_num_tokens = 0
for document in ten_documents:
    encoded_ids = tok.encode(document)
    num_bytes = len(document.encode("utf-8"))

    total_num_bytes += num_bytes
    total_num_tokens += len(encoded_ids)

# print compression ratio
print(total_num_bytes / total_num_tokens)

# PART B

# encode datasets
encoded_ids_train = tok.encode(train_text)
encoded_ids_valid = tok.encode(valid_text)

# serialize into numpy array of type uint16
train_array = np.array(encoded_ids_train, dtype=np.uint16)
valid_array = np.array(encoded_ids_valid, dtype=np.uint16)

os.makedirs('data/numpy', exist_ok=True)
np.save("data/numpy/train_ids.npy", train_array)
np.save("data/numpy/valid_ids.npy", valid_array)