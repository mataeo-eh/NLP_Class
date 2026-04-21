from gtts import gTTS
from playsound3 import playsound

# Read text from file
with open("transcript.txt", "r") as f:
    text = f.read()

# Convert to speech and save
tts = gTTS(text=text, lang="en")
tts.save("output.mp3")

# Play it
playsound("output.mp3")