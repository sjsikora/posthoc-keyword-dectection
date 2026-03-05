SAMPLE_RATE = 16000 # Whisper strictly requires 16kHz audio
CHANNELS = 1 # Mono audio
BLOCK_SIZE = SAMPLE_RATE * 3 # Process the audio in three second chunks

"""
Keywords should only be one single word. It is un defined behavior (and unwise in general)
to have contractions (like I'm or don't) in the keywords.
"""
KEYWORDS = [
    "pineapple",
    "waffle iron",
    "lavender",
    "beam",
    "rock"
]
