from datasets import load_dataset
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from datetime import datetime
import json

# 1. Labels
labels = ["World", "Sports", "Business", "Sci/Tech"]
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# 2. System prompt
SYSTEM_PROMPT = """You are a text classification assistant specializing in news categorization.
Your task is to classify news articles into one of the following categories:
World, Sports, Business, or Sci/Tech.

Respond with ONLY the category name.
Do not provide explanations or additional text.
"""

INCLUDE_SYSTEM_PROMPT = True

# 3. Load dataset
dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].shuffle(seed=42).select(range(2))

# 4. Select FEW-SHOT examples (1 per class, from train set)
few_shot_examples = []
seen_labels = set()

for item in dataset["train"]:
    label = item["label"]
    if label not in seen_labels:
        few_shot_examples.append({
            "text": item["text"],
            "label": label_map[label]
        })
        seen_labels.add(label)
    if len(seen_labels) == 4:
        break

# 5. Few-shot prompt function
def few_shot_prompt(text):
    examples_str = ""
    for ex in few_shot_examples:
        examples_str += f"""
Text:
{ex['text']}
Label: {ex['label']}
"""

    labels_str = ", ".join(labels)

    return f"""
You are a news classification system.

Task:
Classify news articles into one of the following categories: {labels_str}.

Here are some labeled examples:
{examples_str}

Now classify the following article.

Text:
{text}

Answer with only the category name.
"""

# 6. Model
model_name = "llama3.2:3b"

y_true = []
y_pred = []

print(f"Running FEW-SHOT classification with model: {model_name}\n")

# 7. Run classification
for item in test_data:
    prompt = few_shot_prompt(item["text"])

    messages = [
        msg for msg in [
            {"role": "system", "content": SYSTEM_PROMPT} if INCLUDE_SYSTEM_PROMPT else None,
            {"role": "user", "content": prompt}
        ] if msg is not None
    ]

    response = ollama.chat(
        model=model_name,
        messages=messages,
    )
    print(f"Chat response: {response.message}\n")
    prediction = response["message"]["content"].strip()
    pred_label = None
    pred_lower = prediction.lower()

    if "world" in pred_lower:
        pred_label = "World"
    elif "sports" in pred_lower:
        pred_label = "Sports"
    elif "business" in pred_lower:
        pred_label = "Business"
    elif "sci" in pred_lower or "tech" in pred_lower:
        pred_label = "Sci/Tech"
    else:
        pred_label = "Unknown"

    y_true.append(label_map[item["label"]])
    y_pred.append(pred_label)

# 8. Filter invalid predictions
filtered = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
y_true_f, y_pred_f = zip(*filtered)

# 9. Metrics
accuracy = accuracy_score(y_true_f, y_pred_f)
precision = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0)
recall = recall_score(y_true_f, y_pred_f, average="macro", zero_division=0)
f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)
report = classification_report(y_true_f, y_pred_f, zero_division=0)

print("Classification Report:\n", report)

# 10. Save results
os.makedirs("../results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"few_shot_{model_name.replace(':', '_')}_{timestamp}"

results_data = {
    "metadata": {
        "model": model_name,
        "method": "few-shot",
        "timestamp": timestamp,
        "total_samples": len(y_true),
        "valid_predictions": len(y_true_f),
        "unknown_predictions": len(y_true) - len(y_true_f)
    },
    "metrics": {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    },
    "predictions": [
        {"true_label": t, "predicted_label": p}
        for t, p in zip(y_true, y_pred)
    ]
}

results_json = f"../results/{base_filename}.json"
with open(results_json, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\nResults saved to {results_json}")

report_file = f"../results/{base_filename}_report.txt"
with open(report_file, 'w') as f:
    f.write(f"Model: {model_name}\n")
    #f.write(f"Prompt Choice: {PROMPT_CHOICE}\n")
    f.write(f"Include System Prompt: {INCLUDE_SYSTEM_PROMPT}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"\n{'='*60}\n")
    f.write(f"CLASSIFICATION REPORT\n")
    f.write(f"{'='*60}\n\n")
    f.write(report)
    f.write(f"\n\n{'='*60}\n")
    f.write(f"SUMMARY METRICS\n")
    f.write(f"{'='*60}\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"\nTotal samples: {len(y_true)}\n")
    f.write(f"Valid predictions: {len(y_true_f)}\n")
    f.write(f"Unknown predictions: {len(y_true) - len(y_true_f)}\n")
print(f"Report saved to {report_file}\n")