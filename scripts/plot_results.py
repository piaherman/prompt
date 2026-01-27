import json
import matplotlib.pyplot as plt

def load_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["metrics"]["accuracy"], data["metrics"]["f1_score"]

# 🔧 UPDATE PATHS if needed
files = {
    "ZS Simple w/o SP": "../results/zero_shot_llama3.2_3b_simple_20260123_191635.json",
    "ZS Simple w/ SP": "../results/zero_shot_llama3.2_3b_simple_20260122_192523.json",
    "ZS Intermediate w/o SP": "../results/zero_shot_llama3.2_3b_intermediate_20260123_192229.json",
    "ZS Intermediate w/ SP": "../results/zero_shot_llama3.2_3b_intermediate_20260122_194435.json",
    "ZS Advanced w/o SP": "../results/zero_shot_llama3.2_3b_advanced_20260123_192838.json",
    "ZS Advanced w/ SP": "../results/zero_shot_llama3.2_3b_advanced_20260122_195204.json",
    "FS w/o SP": "../results/few_shot_llama3.2_3b_20260123_193647.json",
    "FS w/ SP": "../results/few_shot_llama3.2_3b_20260122_200050.json",
}

methods = []
accuracy = []
f1_scores = []

for method, path in files.items():
    acc, f1 = load_metrics(path)
    methods.append(method)
    accuracy.append(acc)
    f1_scores.append(f1)

# =========================
# Accuracy graph
# =========================
plt.figure()
plt.bar(methods, accuracy)
plt.title("Accuracy by Prompting Strategy")
plt.xlabel("Method")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=90, ha="center")
plt.tight_layout()
plt.savefig("../results/accuracy_prompting_strategies.png", dpi=300)
plt.show()

# =========================
# F1-score graph
# =========================
plt.figure()
plt.bar(methods, f1_scores)
plt.title("Macro F1-score by Prompting Strategy")
plt.xlabel("Method")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=90, ha="center")
plt.tight_layout()
plt.savefig("../results/f1_prompting_strategies.png", dpi=300)
plt.show()
