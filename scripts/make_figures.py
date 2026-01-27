import json
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics"]["accuracy"], data["metrics"]["f1_score"]

# =========================
# Files organized by method and system prompt
# =========================
files = {
    "ZS Simple": {
        "Sys": "../results/zero_shot_llama3.2_3b_simple_20260122_192523.json",
        "NoSys": "../results/zero_shot_llama3.2_3b_simple_20260123_191635.json",
    },
    "ZS Intermediate": {
        "Sys": "../results/zero_shot_llama3.2_3b_intermediate_20260122_194435.json",
        "NoSys": "../results/zero_shot_llama3.2_3b_intermediate_20260123_192229.json",
    },
    "ZS Advanced": {
        "Sys": "../results/zero_shot_llama3.2_3b_advanced_20260122_195204.json",
        "NoSys": "../results/zero_shot_llama3.2_3b_advanced_20260123_192838.json",
    },
    "Few-shot": {
        "Sys": "../results/few_shot_llama3.2_3b_20260122_200050.json",
        "NoSys": "../results/few_shot_llama3.2_3b_20260123_193647.json",
    }
}

labels, accs, f1s = [], [], []

for method, sys_dict in files.items():
    for sys_flag, path in sys_dict.items():
        acc, f1 = load_metrics(path)
        labels.append(f"{method} ({sys_flag})")
        accs.append(acc)
        f1s.append(f1)

# =========================
# Accuracy Figure
# =========================
plt.figure(figsize=(10, 5))
plt.bar(labels, accs)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.title("Accuracy by Prompting Strategy and System Prompt")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("../results/figure1_accuracy_sys_nosys.png", dpi=300)
plt.show()

# =========================
# F1-score Figure
# =========================
plt.figure(figsize=(10, 5))
plt.bar(labels, f1s)
plt.xticks(rotation=90)
plt.ylabel("Macro F1-score")
plt.title("Macro F1-score by Prompting Strategy and System Prompt")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("../results/figure2_f1_sys_nosys.png", dpi=300)
plt.show()
