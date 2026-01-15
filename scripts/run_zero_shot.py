from datasets import load_dataset
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# 1. Labele
labels = ["World", "Sports", "Business", "Sci/Tech"]
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# 2. Zero-shot prompt
def zero_shot_prompt(text):
    labels_str = ", ".join(labels)
    return f"""
Classify the text into the following labels: {labels_str}.

Text:
{text}

Answer with only the label name.
"""

# 3. Naloži podatke
dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].shuffle(seed=42).select(range(10))

# 4. Model
model_name = "llama3.2:3b"

y_true = []
y_pred = []

print(f"Running zero-shot classification with model: {model_name}\n")

# 5. Zagon
for item in test_data:
    prompt = zero_shot_prompt(item["text"])
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],

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

# 7. Metrike
accuracy = accuracy_score(y_true_f, y_pred_f)
precision = precision_score(y_true_f, y_pred_f, average="macro")
recall = recall_score(y_true_f, y_pred_f, average="macro")
f1 = f1_score(y_true_f, y_pred_f, average="macro")

print("ZERO-SHOT RESULTS")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

# 8. Shrani rezultate
results_df = pd.DataFrame({
    "true_label": y_true_f,
    "predicted_label": y_pred_f
})
print(y_true)
print(y_pred)
results_df.to_csv("../results/zero_shot_ministral-3:3b.csv", index=False)
print("\nResults saved to results/zero_shot_ministral-3:3b.csv")
