from datasets import load_dataset
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import os
from datetime import datetime
import json

# 1. Labele
labels = ["World", "Sports", "Business", "Sci/Tech"]
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# 2. System in User prompti
SYSTEM_PROMPT = """You are a text classification assistant specializing in news categorization. Your task is to accurately classify news articles into one of four categories: World, Sports, Business, or Sci/Tech.

Guidelines:
- World: International news, politics, global events, conflicts, diplomacy
- Sports: Athletic events, games, sports figures, competitions
- Business: Finance, economy, markets, companies, trade
- Sci/Tech: Science, technology, innovation, research, computing

Respond with ONLY the category name. Do not provide explanations or additional text."""

PROMPTS = {
  "simple": """Classify the text into the following labels: {labels}.

Text:
{text}

Answer with only the label name.""",
  "intermediate": """You are a news classification system.

Classify the following news article into exactly one of these categories:{labels}.

Text:
{text}

Answer with only the category name.""",
  "advanced": """You are an expert news classifier.

Classify the following text into one of the categories below:

- World: international relations, politics, diplomacy, global events
- Sports: competitive physical activities, teams, matches, tournaments
- Business: companies, markets, finance, economy, stocks
- Sci/Tech: science, technology, research, innovation

Text:
{text}

Answer with only the category name."""
}

# Izberi prompt
# Opcije: "simple", "intermediate", "advanced"
PROMPT_CHOICE = "simple"
# Nastavi na False, če nočeš vključiti system prompta
INCLUDE_SYSTEM_PROMPT = True

# 3. Zero-shot prompt funkcija
def zero_shot_prompt(text):
    labels_str = ", ".join(labels)
    return PROMPTS[PROMPT_CHOICE].format(labels=labels_str, text=text)

# 3. Naloži podatke
dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].shuffle(seed=42).select(range(100))

# 4. Model
# gemma3:4b, ministral-3:3b, llama3.2:3b
model_name = "llama3.2:3b"

y_true = []
y_pred = []

print(f"Running zero-shot classification with model: {model_name}\n")

# 5. Zagon
for item in test_data:
    prompt = zero_shot_prompt(item["text"])

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

    # Normalizacija odgovora
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

# 6. Izločimo neveljavne napovedi
filtered = [(t, p) for t, p in zip(y_true, y_pred) if p != "Unknown"]
y_true_f, y_pred_f = zip(*filtered)

# 7. Matrike
report = classification_report(y_true_f, y_pred_f)
accuracy = accuracy_score(y_true_f, y_pred_f)
precision = precision_score(y_true_f, y_pred_f, average="macro")
recall = recall_score(y_true_f, y_pred_f, average="macro")
f1 = f1_score(y_true_f, y_pred_f, average="macro")
print("Classification Report:\n", report)

# 8. Shrani rezultate
# Ustvari mapo results, če ne obstaja
os.makedirs("../results", exist_ok=True)

# Ime datoteke s časovnim žigom
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"zero_shot_{model_name.replace(':', '_')}_{PROMPT_CHOICE}_{timestamp}"


# 8.1 Shrani podrobne napovedi in metrike (JSON)
results_data = {
    "metadata": {
        "model": model_name,
        "prompt_choice": PROMPT_CHOICE,
        "include_system_prompt": INCLUDE_SYSTEM_PROMPT,
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

results_json_file = f"../results/{base_filename}.json"
with open(results_json_file, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"\nResults saved to {results_json_file}")

# 8.2 Shrani poročilo o klasifikaciji (TXT)
report_file = f"../results/{base_filename}_report.txt"
with open(report_file, 'w') as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Prompt Choice: {PROMPT_CHOICE}\n")
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
