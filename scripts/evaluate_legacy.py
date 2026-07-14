from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

LABEL_MAP = {
    0: "greeting",
    1: "how_are_you",
    2: "name",
    3: "age",
    4: "joke",
    5: "bye",
    6: "thanks",
    7: "help",
    8: "study",
    9: "food",
    10: "weather",
    11: "ai"
}

classifier = pipeline("text-classification", model="./model", tokenizer="./model")

test_data = [
    ("வணக்கம்", 0),
    ("hello bro", 0),
    ("vanakkam machi", 0),

    ("எப்படி இருக்கீங்க", 1),
    ("how are you", 1),
    ("epdi irukinga", 1),

    ("உங்கள் பெயர் என்ன", 2),
    ("what is your name", 2),
    ("un peyar enna", 2),

    ("வயது என்ன", 3),
    ("how old are you", 3),
    ("vayasu enna", 3),

    ("ஒரு ஜோக் சொல்லு", 4),
    ("tell me a joke", 4),
    ("joke sollu", 4),

    ("பிரியாவிடை", 5),
    ("bye", 5),
    ("poitu varen", 5),

    ("நன்றி", 6),
    ("thank you", 6),
    ("romba nandri", 6),

    ("உதவி வேண்டும்", 7),
    ("help me", 7),
    ("udhavi venum", 7),

    ("படிப்பு பற்றி சொல்லுங்கள்", 8),
    ("exam tips", 8),
    ("padippu pathi sollu", 8),

    ("எனக்கு பசி", 9),
    ("food please", 9),
    ("saapadu pathi sollu", 9),

    ("வானிலை எப்படி", 10),
    ("how is the weather", 10),
    ("weather epdi", 10),

    ("AI என்ன", 11),
    ("what is ai", 11),
    ("ai na enna", 11)
]

y_true = []
y_pred = []

print("\nPredictions:\n")

for text, true_label in test_data:
    result = classifier(text)
    pred_label = int(result[0]["label"].split("_")[-1])

    y_true.append(true_label)
    y_pred.append(pred_label)

    print(f"Text: {text}")
    print(f"True: {LABEL_MAP[true_label]} | Predicted: {LABEL_MAP[pred_label]}")
    print("-" * 50)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

print("\nEvaluation Metrics:\n")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)