import random
import json

INTENTS = {
    0: [
        "வணக்கம்", "ஹலோ", "வாங்க", "hello", "hi", "hey",
        "vanakkam", "helo", "hi da", "hello bro"
    ],
    1: [
        "எப்படி இருக்கிறீர்கள்", "எப்படி இருக்கீங்க", "நீங்கள் நலமா",
        "how are you", "how r u", "are you fine",
        "epdi irukinga", "epdi iruka", "eppadi irukkinga"
    ],
    2: [
        "உன் பெயர் என்ன", "உங்கள் பெயர் என்ன", "பெயர் என்ன",
        "what is your name", "your name", "who are you",
        "un peyar enna", "unga name enna", "nee yaaru"
    ],
    3: [
        "உங்கள் வயது என்ன", "வயது என்ன", "உனக்கு வயசு என்ன",
        "how old are you", "what is your age",
        "vayasu enna", "unakku vayasu evlo"
    ],
    4: [
        "ஒரு ஜோக் சொல்லு", "ஜோக் சொல்லுங்கள்", "என்னை சிரிக்க வையுங்கள்",
        "tell me a joke", "say a joke", "make me laugh",
        "oru joke sollu", "joke sollu", "funny ah sollu"
    ],
    5: [
        "பிரியாவிடை", "போயிட்டு வரேன்", "நான் போறேன்",
        "bye", "goodbye", "see you later",
        "poitu varen", "naan poren", "bye da"
    ],
    6: [
        "நன்றி", "மிக்க நன்றி", "ரொம்ப நன்றி",
        "thanks", "thank you", "many thanks",
        "nandri", "romba nandri", "thanks bro"
    ],
    7: [
        "உதவி வேண்டும்", "என்ன உதவி செய்ய முடியும்", "எனக்கு உதவி செய்",
        "help", "can you help me", "what can you do",
        "udhavi venum", "enna help panna mudiyum", "help pannunga"
    ],
    8: [
        "படிப்பு பற்றி சொல்லுங்கள்", "எப்படி நல்லா படிக்கலாம்", "தேர்வுக்கு எப்படி தயாராகலாம்",
        "tell me about study", "how to study well", "exam tips",
        "padippu pathi sollu", "nalla padikka tips", "exam ku epdi prepare panrathu"
    ],
    9: [
        "சாப்பாடு பற்றி சொல்லுங்கள்", "எனக்கு பசி", "தமிழ் உணவு என்ன",
        "tell me about food", "i am hungry", "what food do you like",
        "saapadu pathi sollu", "enakku pasi", "enna saapadu nalla irukkum"
    ],
    10: [
        "வானிலை எப்படி", "மழை பெய்யுமா", "இன்று வெயில் அதிகமா",
        "how is the weather", "is it raining", "is it sunny today",
        "weather epdi", "mazhai varuma", "veyil jasthi ah"
    ],
    11: [
        "AI என்ன", "செயற்கை நுண்ணறிவு என்றால் என்ன", "chatbot என்றால் என்ன",
        "what is ai", "what is artificial intelligence", "what is a chatbot",
        "ai na enna", "artificial intelligence na enna", "chatbot na enna"
    ]
}

NOISE_WORDS = [
    "da", "bro", "machi", "pls", "boss", "yo", "hey", "please", "friend"
]

def augment(text: str) -> str:
    text = text.strip()

    # random suffix
    if random.random() > 0.5:
        text = text + " " + random.choice(NOISE_WORDS)

    # random prefix
    if random.random() > 0.7:
        text = random.choice(NOISE_WORDS) + " " + text

    # random punctuation
    if random.random() > 0.7:
        text += random.choice(["!", "!!", "...", "?"])

    return text

def build_dataset(samples_per_class: int = 700):
    dataset = []

    for label, phrases in INTENTS.items():
        for _ in range(samples_per_class):
            text = random.choice(phrases)
            text = augment(text)
            dataset.append({
                "text": text,
                "label": label
            })

    random.shuffle(dataset)
    return dataset

def save_jsonl(dataset, filename="intent_dataset.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    dataset = build_dataset(samples_per_class=700)
    save_jsonl(dataset)
    print("Dataset created successfully: intent_dataset.jsonl")
    print(f"Total samples: {len(dataset)}")