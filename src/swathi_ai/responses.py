from __future__ import annotations

RESPONSE_BANK = {
    "greeting": {"tamil": "வணக்கம்! 🌺 நான் Swathi AI. உங்களுக்கு எப்படி உதவலாம்?", "tanglish": "Vanakkam! 🌺 Naan Swathi AI. Ungalukku eppadi udhavalam?", "english": "Hello! 🌺 I'm Swathi AI. How can I help you?"},
    "how_are_you": {"tamil": "நான் நன்றாக இருக்கிறேன் 😊 நீங்கள் எப்படி இருக்கிறீர்கள்?", "tanglish": "Naan nandraaga irukkiren 😊 Neenga eppadi irukkinga?", "english": "I'm doing well 😊 How are you?"},
    "name": {"tamil": "என் பெயர் Swathi AI 🌺", "tanglish": "En peyar Swathi AI 🌺", "english": "My name is Swathi AI 🌺"},
    "age": {"tamil": "நான் ஒரு AI chatbot, மனிதர்களைப் போல எனக்கு வயது இல்லை 😊", "tanglish": "Naan oru AI chatbot; manidhargal pola enakku vayasu illai 😊", "english": "I'm an AI chatbot, so I don't have an age like humans 😊"},
    "joke": {"tamil": "😄 ஆசிரியர்: ஏன் தாமதம்? மாணவன்: சார், நேரம் தான் வேகமாக போய்விட்டது!", "tanglish": "😄 Aasiriyar: Yen thaamatham? Maanavan: Saar, neram dhaan vegama poiduchu!", "english": "😄 Teacher: Why are you late? Student: Sir, time moved too fast!"},
    "bye": {"tamil": "பிரியாவிடை 👋 மீண்டும் வாருங்கள்!", "tanglish": "Priyavidai 👋 Meendum vaanga!", "english": "Goodbye 👋 See you again!"},
    "thanks": {"tamil": "நன்றி! 😊", "tanglish": "Nandri! 😊", "english": "You're welcome! 😊"},
    "help": {"tamil": "தமிழ், Tanglish அல்லது English-ல் பேசலாம். படிப்பு, உணவு, வானிலை, இசை, திரைப்படம் மற்றும் AI பற்றி கேளுங்கள்.", "tanglish": "Tamil, Tanglish illa English-la pesalaam. Study, food, weather, music, movie, AI pathi kelunga.", "english": "You can chat in Tamil, Tanglish, or English and ask about study, food, weather, music, movies, or AI."},
    "study": {"tamil": "தினமும் ஒரு சிறிய திட்டத்துடன் படித்து, இடைவெளியில் மீள்பார்வை செய்யுங்கள் 📚", "tanglish": "Dhinamum oru siriya plan-oda padichu, intervals-la revision pannunga 📚", "english": "Study with a small daily plan and revise at spaced intervals 📚"},
    "food": {"tamil": "தமிழ் உணவில் இட்லி, தோசை, சாம்பார் மற்றும் பிரியாணி மிகவும் பிரபலமானவை 🍛", "tanglish": "Tamil unavula idli, dosa, sambar, biryani romba popular 🍛", "english": "Idli, dosa, sambar, and biryani are popular Tamil foods 🍛"},
    "weather": {"tamil": "நேரடி வானிலை தகவலுக்கு இணைய சேவை தேவை. உங்கள் நகரத்தைச் சொல்லுங்கள்.", "tanglish": "Live weather-ku internet service thevai. Unga city-a sollunga.", "english": "Live weather requires an internet service. Tell me your city."},
    "movie": {"tamil": "தமிழ் திரைப்படங்கள் இசை, உணர்ச்சி மற்றும் கதைக்காகப் பிரபலமானவை 🎬", "tanglish": "Tamil movies isai, unarchi, kadhai-kaaga popular 🎬", "english": "Tamil movies are known for music, emotion, and storytelling 🎬"},
    "music": {"tamil": "தமிழ் இசை உலகம் மிகவும் வளமானது 🎵", "tanglish": "Tamil isai ulagam romba azhagaanadhu 🎵", "english": "Tamil music has a rich and beautiful tradition 🎵"},
    "cricket": {"tamil": "கிரிக்கெட் பற்றி பேசலாம் 🏏 நேரடி score-க்கு இணைய இணைப்பு தேவை.", "tanglish": "Cricket pathi pesalaam 🏏 Live score-ku internet connection thevai.", "english": "We can talk about cricket 🏏 Live scores require an internet connection."},
    "ai": {"tamil": "AI என்பது கற்றல், மொழி புரிதல் மற்றும் முடிவெடுப்பை கணினிகளால் செய்ய உதவும் தொழில்நுட்பம்.", "tanglish": "AI-na computers learn panna, language purinjukka, decisions edukka help pannura technology.", "english": "AI enables computers to learn, understand language, and support decisions."},
    "offline": {"tamil": "நான் இப்போது offline mode-ல் இருக்கிறேன். ஆதரிக்கப்படும் கேள்விகளில் ஒன்றைக் கேளுங்கள்.", "tanglish": "Naan ippo offline mode-la irukken. Supported topic pathi kelunga.", "english": "I'm in offline mode. Ask about one of the supported topics."},
    "fallback": {"tamil": "மன்னிக்கவும், அதை முழுமையாகப் புரிந்துகொள்ளவில்லை. வேறு விதமாக கேட்க முடியுமா?", "tanglish": "Mannikkavum, adha full-a purinjukkala. Vera madhiri kekkareengala?", "english": "Sorry, I didn't fully understand that. Could you rephrase it?"},
}


def format_reply(intent: str, language: str, show_all: bool) -> str:
    reply = RESPONSE_BANK.get(intent, RESPONSE_BANK["fallback"])
    if not show_all:
        return reply.get(language, reply["english"])
    ordered = ("english", "tamil", "tanglish") if language == "english" else ("tamil", "tanglish", "english")
    labels = {"tamil": "Tamil", "tanglish": "Tanglish", "english": "English"}
    return "\n\n".join(f"**{labels[key]}:**\n{reply[key]}" for key in ordered)
