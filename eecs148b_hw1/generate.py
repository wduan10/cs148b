import argparse
import pickle
from pathlib import Path
from typing import Iterable, Iterator

import regex as re
import torch

from eecs148b_hw1 import transformer_lm, tokenizer, decoding


def get_torch_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_model_from_checkpoint(ckpt: dict, device: str):
    if "args" not in ckpt:
        raise ValueError("Checkpoint does not contain saved training args.")

    args = ckpt["args"]
    model_dtype = get_torch_dtype(args.get("torch_dtype", "float32"))

    model = transformer_lm.TransformerLanguageModel(
        vocab_size=args["vocab_size"],
        context_length=args["context_length"],
        d_model=args["d_model"],
        num_heads=args["num_heads"],
        d_ff=args["d_ff"],
        num_layers=args["num_layers"],
        device=device,
        dtype=model_dtype,
    ).to(device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str] | None):
    tok = tokenizer.Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    end_of_text_token_id = None
    if special_tokens and "<|endoftext|>" in special_tokens:
        end_of_text_token_id = tok.vocab_swapped.get(b"<|endoftext|>")

    return tok, end_of_text_token_id


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.pkl")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-file", type=str, default="generated_text.txt")
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help='Optional special tokens, e.g. --special-tokens "<|endoftext|>"',
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    device = cli_args.device

    ckpt = torch.load(cli_args.checkpoint, map_location=device)
    model, train_args = build_model_from_checkpoint(ckpt, device)

    tokenizer, end_of_text_token_id = load_tokenizer(
        vocab_path=cli_args.vocab,
        merges_path=cli_args.merges,
        special_tokens=cli_args.special_tokens,
    )

    prompt_ids = tokenizer.encode(cli_args.prompt)
    prompt = torch.tensor(prompt_ids, dtype=torch.long)

    generated_ids = decoding.decode(
        model=model,
        prompt=prompt,
        max_new_tokens=cli_args.max_new_tokens,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        end_of_text_token_id=end_of_text_token_id,
    )

    generated_text = tokenizer.decode(generated_ids.tolist())

    Path(cli_args.output_file).write_text(generated_text, encoding="utf-8")

    print("=== GENERATED TEXT ===")
    print(generated_text)
    print()
    print(f"Saved generated text to: {cli_args.output_file}")
    print(f"Generated {len(generated_ids) - len(prompt_ids)} new tokens.")


if __name__ == "__main__":
    main()