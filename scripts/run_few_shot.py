from datasets import load_dataset
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

labels = ["World", "Sports", "Business", "Sci/Tech"]
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

few_shot_examples = [
    {"text": "The government announced new foreign policy measures.", "label": "World"},
    {"text": "The team won the championship after a thrilling final.", "label": "Sports"},
    {"text": "Company shares rose after the quarterly earnings report.", "label": "Business"},
    {"text": "Researchers developed a new artificial intelligence system.", "label": "Sci/Tech"},
]

def few_shot_prompt(text):
    examples_str = ""
    for ex in few_shot_examples:
        examples_str += f"Text: {ex['text']}\nLabel: {ex['label']}\n\n"

    labels_str = ", ".join(labels)
    return f"""
Classify the text into the following labels: {labels_str}.

Examples:
{examples_str}

Text:
{text}

Answer with only the label name.
"""

dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].shuffle(seed=42).select(range(30))  # 30 je dovolj

MODEL_NAME = "llama3.2:3b"  # hiter za začetek

y_true, y_pred = [], []

for item in test_data:
    prompt = few_shot_prompt(item["text"])
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        #options={"num_predict": 5, "temperature": 0.0}
    )

    out = response["message"]["content"].lower()
    if "world" in out:
        pred = "World"
    elif "sports" in out:
        pred = "Sports"
    elif "business" in out:
        pred = "Business"
    elif "sci" in out or "tech" in out:
        pred = "Sci/Tech"
    else:
        pred = "Unknown"

    y_true.append(label_map[item["label"]])
    y_pred.append(pred)

# odstrani Unknown
filtered = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
y_true_f, y_pred_f = zip(*filtered)

acc = accuracy_score(y_true_f, y_pred_f)
prec = precision_score(y_true_f, y_pred_f, average="macro")
rec = recall_score(y_true_f, y_pred_f, average="macro")
f1 = f1_score(y_true_f, y_pred_f, average="macro")

print("FEW-SHOT RESULTS")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

pd.DataFrame({"true": y_true_f, "pred": y_pred_f}).to_csv(
    "../results/few_shot_llama.csv", index=False
)
