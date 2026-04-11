import train_bpe
import pickle
import os
import sys

train = len(sys.argv) > 1 and sys.argv[1].lower() == 'train'
test = len(sys.argv) > 1 and sys.argv[1].lower() == 'test'

if train:
    os.makedirs("data/pickle", exist_ok=True)

    input_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']

    print('Training BPE on TinyStories file.')
    vocab, merges = train_bpe.train_bpe(input_path, vocab_size, special_tokens)
    print('Done training.')

    pickle.dump(vocab, open('data/pickle/tinystories_vocab', 'wb'))
    pickle.dump(merges, open('data/pickle/tinystories_merges', 'wb'))
elif test:
    vocab = pickle.load(open('data/pickle/tinystories_vocab', 'rb'))
    _, longest_token = max(
        vocab.items(),
        key=lambda x: len(x[1])
    )

    print(longest_token)