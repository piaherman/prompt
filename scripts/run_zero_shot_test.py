from datasets import load_dataset
import ollama

# 1. Definiramo labele
labels = ["World", "Sports", "Business", "Sci/Tech"]

# 2. Zero-shot prompt
def zero_shot_prompt(text):
    labels_str = ", ".join(labels)
    return f"""
Classify the text into the following labels: {labels_str}.
Text:
{text}
Answer with only the label name.
"""

# 3. Naložimo AG News (samo nekaj primerov)
dataset = load_dataset("fancyzhx/ag_news")
test_data = dataset["test"].select(range(5))

# 4. Izberemo model
model_name = "ministral-3:3b"
print(f"Running zero-shot classification with model: {model_name}\n")

# 5. Zagon eksperimenta
for i, item in enumerate(test_data):
    text = item["text"]
    true_label = labels[item["label"]]

    prompt = zero_shot_prompt(text)

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    prediction = response["message"]["content"].strip()

    print(f"Example {i+1}")
    print("True label:     ", true_label)
    print("Model prediction:", prediction)
    print("-" * 40)
