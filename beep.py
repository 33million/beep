import io
import os
import datetime
import requests
import pygame
import anthropic
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"/automations/.env")

elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]
SAMPLE_RATE = 44100

# --- Speak Function ---
def speak(text):
    print(f"\nPriyanka: {text}")
    audio_stream = elevenlabs_client.text_to_speech.stream(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings={
            "stability": 0.6,
            "similarity_boost": 0.8,
            "speed": 0.83
        }
    )
    audio_bytes = b"".join(chunk for chunk in audio_stream)
    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
    pygame.mixer.music.load(io.BytesIO(audio_bytes))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

# --- Listen Function ---
def listen(silence_threshold=500, silence_duration=2.0, max_duration=30):
    print("\nListening...")
    chunk_size = int(SAMPLE_RATE * 0.1)
    audio_chunks = []
    silent_chunks = 0
    silent_chunks_needed = int(silence_duration / 0.1)
    max_chunks = int(max_duration / 0.1)
    speaking_started = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while len(audio_chunks) < max_chunks:
            chunk, _ = stream.read(chunk_size)
            audio_chunks.append(chunk.copy())
            volume = np.abs(chunk).mean()

            if volume > silence_threshold:
                speaking_started = True
                silent_chunks = 0
            elif speaking_started:
                silent_chunks += 1
                if silent_chunks >= silent_chunks_needed:
                    break

    recording = np.concatenate(audio_chunks, axis=0)

    # 1. Bell tone immediately

    tone_duration = 1.2
    t = np.linspace(0, tone_duration, int(SAMPLE_RATE * tone_duration), False)

    # C major 9
    root = 523.25  # C5
    maj3 = 659.25  # E5
    p5 = 783.99  # G5
    maj7 = 987.77  # B5
    maj9 = 1174.66  # D6

    bell = np.sin(2 * np.pi * root * t)
    bell += 0.8 * np.sin(2 * np.pi * maj3 * t)
    bell += 0.6 * np.sin(2 * np.pi * p5 * t)
    bell += 0.4 * np.sin(2 * np.pi * maj7 * t)
    bell += 0.25 * np.sin(2 * np.pi * maj9 * t)

    fade = np.exp(-4 * t)
    bell = (bell * fade * 0.12 * 32767).astype(np.int16)
    bell_stereo = np.column_stack((bell, bell))

    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    bell_sound = pygame.sndarray.make_sound(bell_stereo)
    bell_sound.play()
    pygame.time.wait(2000)

    # 2. Then processing
    print("Processing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, recording)
        temp_path = f.name

    with open(temp_path, "rb") as audio_file:
        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v1",
            language_code="en"
        )

    os.unlink(temp_path)
    text = transcription.text.strip()
    print(f"Patrick: {text}")
    return text

# --- Get Date ---
now = datetime.datetime.now()
date_string = now.strftime("%A, %B %d, %Y")

# --- Get Weather for Philomath, OR ---
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 44.5407,
    "longitude": -123.3674,
    "current": "temperature_2m,weathercode,windspeed_10m",
    "daily": "temperature_2m_max,temperature_2m_min,weathercode,sunrise,sunset",
    "temperature_unit": "fahrenheit",
    "windspeed_unit": "mph",
    "timezone": "America/Los_Angeles"
}
weather_response = requests.get(url, params=params)
weather_data = weather_response.json()

temp = weather_data["current"]["temperature_2m"]
windspeed = weather_data["current"]["windspeed_10m"]
weathercode = weather_data["current"]["weathercode"]
temp_max = weather_data["daily"]["temperature_2m_max"][0]
temp_min = weather_data["daily"]["temperature_2m_min"][0]
daily_weathercode = weather_data["daily"]["weathercode"][0]
sunrise_raw = weather_data["daily"]["sunrise"][0]
sunset_raw = weather_data["daily"]["sunset"][0]
sunrise = datetime.datetime.fromisoformat(sunrise_raw).strftime("%I:%M %p")
sunset = datetime.datetime.fromisoformat(sunset_raw).strftime("%I:%M %p")

def interpret_weather(code):
    if code == 0:
        return "clear skies"
    elif code in [1, 2, 3]:
        return "partly cloudy"
    elif code in [45, 48]:
        return "foggy"
    elif code in [51, 53, 55]:
        return "drizzling"
    elif code in [61, 63, 65]:
        return "rainy"
    elif code in [71, 73, 75]:
        return "snowing"
    elif code in [80, 81, 82]:
        return "showery"
    elif code in [95, 96, 99]:
        return "stormy"
    else:
        return "mixed conditions"

weather_description = interpret_weather(weathercode)
daily_weather_description = interpret_weather(daily_weathercode)

# --- Generate Morning Message ---
prompt = f"""You are a warm, gentle morning companion speaking softly to Patrick, who is just waking up. Address him by name naturally, but not repeatedly.
Today is {date_string}.
Current weather in Philomath, Oregon: {weather_description}, {temp}°F with winds at {windspeed} mph.
Today's forecast: {daily_weather_description}, high of {temp_max}°F and low of {temp_min}°F.
Sunrise is at {sunrise} and sunset is at {sunset}.

Write a soft, spoken good morning message that flows naturally through these sections:
1. A warm greeting mentioning the current conditions and what the day ahead looks like weather-wise
2. Mention sunrise and sunset times naturally, as if noting the shape of the day
3. An uplifting quote with a brief, gentle reflection
4. One genuinely interesting fact about the world
5. A light brain teaser or thought puzzle to wake the mind. Ask it and then leave it open. Do not give the answer.

Between each section, write <break time="1.5s" /> on its own to create a natural pause.

Write it as one flowing piece meant to be spoken aloud, slowly and warmly.
Keep it under 350 words. Do not use any markdown, bullet points, or special characters except for the break tags."""

message = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

morning_script = message.content[0].text

# --- Speak Morning Message ---
speak(morning_script)

# --- Conversation Loop ---
conversation_history = [
    {
        "role": "assistant",
        "content": morning_script
    }
]

speak("What's on your mind, Patrick?")

while True:
    user_input = listen()

    if not user_input:
        speak("I didn't quite catch that. What were you saying?")
        continue

    farewell_words = ["goodbye", "bye", "that's all", "thank you", "thanks", "stop", "exit"]
    if any(word in user_input.lower() for word in farewell_words):
        speak("Have a wonderful day, Patrick. Go gently.")
        break

    conversation_history.append({"role": "user", "content": user_input})

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a warm, gentle morning companion named Priyanka speaking with Patrick. Keep responses concise, warm, and conversational. Speak as if you are talking, not writing.",
        messages=conversation_history
    )

    reply = response.content[0].text
    conversation_history.append({"role": "assistant", "content": reply})

    speak(reply)