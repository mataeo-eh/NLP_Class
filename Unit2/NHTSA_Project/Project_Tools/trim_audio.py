from pydub import AudioSegment
from pydub.silence import detect_nonsilent

audio = AudioSegment.from_file("New Recording.wav")

# Find chunks where audio is above silence threshold
nonsilent_ranges = detect_nonsilent(
    audio,
    min_silence_len=300,   # silence must be at least 300ms to count
    silence_thresh=-40     # dBFS threshold — adjust if needed
)

# Concatenate only the voiced segments
voiced = AudioSegment.empty()
for start, end in nonsilent_ranges:
    voiced += audio[start:end]
    if len(voiced) >= 10000:  # 10 seconds in ms
        break

clip = voiced[:10000]
clip.export("voice_clip.wav", format="wav")