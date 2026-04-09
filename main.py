import sounddevice as sd
import numpy as np
import queue
import sys
from classes.stt import WhisperModel
from classes.detector import PhoneticKeywordDetector
from config import SAMPLE_RATE, CHANNELS, BLOCK_SIZE, KEYWORDS, PHONETIC_K

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback triggered by sounddevice for every new block of audio."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def choose_microphone():
    """Queries system for input devices and prompts the user to select one."""
    print("🎙️ Available Microphone Inputs:\n" + "-"*30)
    devices = sd.query_devices()
    valid_inputs = []

    # Filter and display only devices that can capture audio
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{i}] {device['name']}")
            valid_inputs.append(i)

    if not valid_inputs:
        print("No input devices found. Please connect a microphone.")
        sys.exit(1)

    # Loop until the user provides a valid device ID
    while True:
        try:
            choice = input("\nSelect the microphone ID you want to use: ")
            device_id = int(choice)
            if device_id in valid_inputs:
                print(f"Selected: {devices[device_id]['name']}\n")
                return device_id
            else:
                print("Invalid ID. Please choose a number from the list above.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    # 1. Prompt user for device selection first
    selected_device = choose_microphone()

    whisper = WhisperModel()
    detector = PhoneticKeywordDetector(KEYWORDS, PHONETIC_K)

    # 2. Open the audio stream using the selected device
    stream = sd.InputStream(
        device=selected_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )

    with stream:
        print("🎤 Listening... (Press Ctrl+C to stop)")
        try:
            while True:
                audio_chunk = audio_queue.get()

                result = whisper.transcribe_audio_chunk(audio_chunk)

                if result:
                    print(f"Transcription: {result.pharse}")
                    detections = detector.detect_keyword(result.pharse)
                    for d in detections.keywords:
                        print(f"  >> KEYWORD '{d.pharse}' detected (confidence: {d.confidence:.2f})")

        except KeyboardInterrupt:
            print("\nStopping transcription pipeline...")

if __name__ == "__main__":
    main()
