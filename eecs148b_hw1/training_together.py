import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from eecs148b_hw1 import (
    transformer_lm,
    cross_entropy_loss,
    data_loading,
    experiment_logger
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM.")

    # Data
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to train token file (.bin/.npy compatible with np.memmap)")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Path to val token file (.bin/.npy compatible with np.memmap)")
    parser.add_argument("--dtype", type=str, default="int32",
                        choices=["int16", "int32", "int64", "uint16", "uint32"],
                        help="dtype used in memmapped token files")

    # Model
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Logging / eval / checkpoint
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume-from", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def get_torch_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_memmap(path: str, dtype: str) -> np.memmap:
    np_dtype = getattr(np, dtype)
    return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def estimate_split_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
):
    model.eval()

    losses = []
    for _ in range(eval_iters):
        x, y = data_loading.get_batch(
            dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(x)  # (B, T, vocab_size)
        loss = cross_entropy_loss.cross_entropy_loss(logits, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    args,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    model_dtype = get_torch_dtype(args.torch_dtype)

    print("Loading datasets with np.memmap...")
    train_data = load_memmap(args.train_data, args.dtype)
    val_data = load_memmap(args.val_data, args.dtype)

    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens:   {len(val_data)}")
    print(f"Using device: {device}")

    logger = experiment_logger.ExperimentLogger(os.path.join(args.checkpoint_dir, "run_log.jsonl"))
    logger.log(
        event="config",
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        device=args.device,
    )

    # Adjust this class name if your implementation uses a different one.
    model = transformer_lm.TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        device=device,
        dtype=model_dtype,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_val_loss = float("inf")

    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print("Starting training...")
    t0 = time.time()

    for step in range(start_step, args.max_iters):
        model.train()

        x, y = data_loading.get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)  # (B, T, vocab_size)
        loss = cross_entropy_loss.cross_entropy_loss(logits, y)
        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            train_ppl = math.exp(loss.item())
            print(
                f"step {step:6d} | "
                f"train loss {loss.item():.4f} | "
                f"train ppl {train_ppl:.4f} | "
                f"time {elapsed:.1f}s"
            )
            logger.log(
                event="train",
                step=step,
                train_loss=loss.item(),
            )

        if step % args.eval_interval == 0:
            train_eval_loss, train_eval_ppl = estimate_split_loss(
                model=model,
                dataset=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )
            val_loss, val_ppl = estimate_split_loss(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )

            print(
                f"[eval @ step {step}] "
                f"train loss {train_eval_loss:.4f}, train ppl {train_eval_ppl:.4f} | "
                f"val loss {val_loss:.4f}, val ppl {val_ppl:.4f}"
            )
            logger.log(
                event="eval",
                step=step,
                train_eval_loss=train_eval_loss,
                val_loss=val_loss,
                train_perplexity=train_eval_ppl,
                val_perplexity=val_ppl,
            )

            latest_ckpt = os.path.join(args.checkpoint_dir, "latest.pt")
            save_checkpoint(
                latest_ckpt,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                args=args,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(args.checkpoint_dir, "best.pt")
                save_checkpoint(
                    best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    args=args,
                )
                print(f"Saved new best checkpoint to {best_ckpt}")

        if step > 0 and step % args.save_interval == 0:
            periodic_ckpt = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            save_checkpoint(
                periodic_ckpt,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                args=args,
            )
            print(f"Saved checkpoint to {periodic_ckpt}")

    final_ckpt = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(
        final_ckpt,
        model=model,
        optimizer=optimizer,
        step=args.max_iters - 1,
        best_val_loss=best_val_loss,
        args=args,
    )
    print(f"Training complete. Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()