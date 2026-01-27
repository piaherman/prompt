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

FEW_SHOT_EXAMPLES = [
    {
        "text": "Venezuelans Vote Early in Referendum on Chavez Rule (Reuters) "
                "Reuters - Venezuelans turned out early and in large numbers on Sunday "
                "to vote in a historic referendum that will either remove left-wing "
                "President Hugo Chavez from office or give him a new mandate to govern "
                "for the next two years.",
        "label": "World"
    },
    {
        "text": "Phelps, Thorpe Advance in 200 Freestyle (AP) "
                "AP - Michael Phelps took care of qualifying for the Olympic "
                "200-meter freestyle semifinals Sunday, and then found out he had been "
                "added to the American team for the evening's 400 freestyle relay final.",
        "label": "Sports"
    },
    {
        "text": "Wall St. Bears Claw Back Into the Black (Reuters) "
                "Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, "
                "are seeing green again.",
        "label": "Business"
    },
    {
        "text": "'Madden,' 'ESPN' Football Score in Different Ways (Reuters) "
                "Reuters - EA Sports would like to think absenteeism was high because "
                "Madden NFL 2005 was released, and fans took time off to play it.",
        "label": "Sci/Tech"
    }
]
# 3. Load dataset
dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].shuffle(seed=42).select(range(100))

# 5. Few-shot prompt function
def few_shot_prompt(text):
    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        examples_str += f"""
Example {i}:
Text:
{ex['text']}
Category: {ex['label']}
"""

    return f"""
You are a news classification system.

Task:
Classify news articles into one of the following categories:
- World
- Sports
- Business
- Sci/Tech

Below are some labeled examples:
{examples_str}

Now classify the following article.

Text:
{text}

Answer with ONLY the category name.
"""

# 6. Model
model_name = "llama3.2:3b"

y_true = []
y_pred = []

print(f"Running few-shot classification with model: {model_name}\n")

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