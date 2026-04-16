import json
import matplotlib.pyplot as plt

train_steps, train_losses = [], []
eval_steps, val_losses = [], []

with open("data/checkpoints_ln_ablation/run_log.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec["event"] == "train" and "train_loss" in rec:
            train_steps.append(rec["step"])
            train_losses.append(rec["train_loss"])
        elif rec["event"] == "eval" and "val_loss" in rec:
            eval_steps.append(rec["step"])
            val_losses.append(rec["val_loss"])

plt.plot(train_steps, train_losses, label="train loss")
plt.plot(eval_steps, val_losses, label="val loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("plots/loss_curve_ln_ablation.png")
