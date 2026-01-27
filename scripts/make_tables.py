import json
import pandas as pd

# -------- Helper function --------
def load_metrics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics"]

# -------- ZERO-SHOT TABLE --------
zero_shot_files = {
    "Simple": {
        "Yes": "../results/zero_shot_llama3.2_3b_simple_20260122_192523.json",
        "No": "../results/zero_shot_llama3.2_3b_simple_20260123_191635.json",
    },
    "Intermediate": {
        "Yes": "../results/zero_shot_llama3.2_3b_intermediate_20260122_194435.json",
        "No": "../results/zero_shot_llama3.2_3b_intermediate_20260123_192229.json",
    },
    "Advanced": {
        "Yes": "../results/zero_shot_llama3.2_3b_advanced_20260122_195204.json",
        "No": "../results/zero_shot_llama3.2_3b_advanced_20260123_192838.json",
    },
}


zs_rows = []

for prompt_type, sys_dict in zero_shot_files.items():
    for sys_flag, path in sys_dict.items():
        m = load_metrics(path)
        zs_rows.append([
            prompt_type,
            sys_flag,
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1_score"]
        ])

df_zero_shot = pd.DataFrame(
    zs_rows,
    columns=["Prompt Type", "System Prompt", "Accuracy", "Precision", "Recall", "F1-score"]
)

df_zero_shot.to_excel("../results/table1_zero_shot.xlsx", index=False)

# -------- FEW-SHOT TABLE --------
few_shot_files = {
    "Few-shot": {
        "Yes": "../results/few_shot_llama3.2_3b_20260122_200050.json",
        "No": "../results/few_shot_llama3.2_3b_20260123_193647.json",
    }
}

fs_rows = []

for method, sys_dict in few_shot_files.items():
    for sys_flag, path in sys_dict.items():
        m = load_metrics(path)
        fs_rows.append([
            method,
            sys_flag,
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1_score"]
        ])

df_few_shot = pd.DataFrame(
    fs_rows,
    columns=["Method", "System Prompt", "Accuracy", "Precision", "Recall", "F1-score"]
)

df_few_shot.to_excel("../results/table2_few_shot.xlsx", index=False)

print("Few-shot table saved to /results/")

