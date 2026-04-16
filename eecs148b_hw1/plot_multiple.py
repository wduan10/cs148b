import json
import matplotlib.pyplot as plt


def load_run(path):
    train_steps, train_losses = [], []
    eval_steps, val_losses = [], []

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["event"] == "train" and "train_loss" in rec:
                train_steps.append(rec["step"])
                train_losses.append(rec["train_loss"])
            elif rec["event"] == "eval" and "val_loss" in rec:
                eval_steps.append(rec["step"])
                val_losses.append(rec["val_loss"])

    return train_steps, train_losses, eval_steps, val_losses


# Load both runs
base_path = "data/checkpoints/run_log.jsonl"
ablation_path = "data/checkpoints_pe_ablation/run_log.jsonl"

base_train_steps, base_train_losses, base_eval_steps, base_val_losses = load_run(
    base_path)
abl_train_steps, abl_train_losses, abl_eval_steps, abl_val_losses = load_run(
    ablation_path)


# Plot
plt.figure(figsize=(8, 5))

# Train curves
plt.plot(base_train_steps, base_train_losses,
         label="train (baseline)", linestyle="--")
plt.plot(abl_train_steps, abl_train_losses,
         label="train (ablation)", linestyle="--")

# Validation curves (usually more important)
plt.plot(base_eval_steps, base_val_losses, label="val (baseline)")
plt.plot(abl_eval_steps, abl_val_losses, label="val (ablation)")

plt.xlabel("step")
plt.ylabel("loss")
plt.title("Baseline vs Positional Embedding Ablation")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("plots/loss_curve_comparison.png")
plt.show()
