import sounddevice as sd
import numpy as np
import queue
import sys
import time
from collections import defaultdict
from classes.stt import WhisperModel
from classes.detector import PhoneticKeywordDetector
from config import (
    SAMPLE_RATE, CHANNELS, WINDOW_SIZE, STEP_SIZE,
    KEYWORDS, PHONETIC_K, DETECTION_COOLDOWN,
)

audio_queue: queue.Queue[np.ndarray] = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def choose_microphone() -> int:
    print("Available Microphone Inputs:\n" + "-" * 30)
    devices = sd.query_devices()
    valid_inputs = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{i}] {device['name']}")
            valid_inputs.append(i)
    if not valid_inputs:
        print("No input devices found. Please connect a microphone.")
        sys.exit(1)
    while True:
        try:
            choice = input("\nSelect the microphone ID you want to use: ")
            device_id = int(choice)
            if device_id in valid_inputs:
                print(f"Selected: {devices[device_id]['name']}\n")
                return device_id
            print("Invalid ID. Please choose a number from the list above.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    selected_device = choose_microphone()

    whisper  = WhisperModel()
    detector = PhoneticKeywordDetector(KEYWORDS, PHONETIC_K)

    # Rolling audio buffer — holds WINDOW_SIZE samples (3 s).
    # The stream fires every STEP_SIZE samples (1 s), giving 67% window overlap.
    # Overlapping windows catch keywords that straddle a chunk boundary.
    window = np.zeros((WINDOW_SIZE, CHANNELS), dtype='float32')
    chunks_received = 0
    chunks_to_fill  = WINDOW_SIZE // STEP_SIZE  # 3 chunks before first analysis

    # Cooldown: suppress re-firing the same keyword across overlapping windows.
    last_detection: dict[str, float] = defaultdict(float)

    stream = sd.InputStream(
        device=selected_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=STEP_SIZE,
        callback=audio_callback,
    )

    with stream:
        print(f"Listening... (window={WINDOW_SIZE // SAMPLE_RATE}s, step={STEP_SIZE // SAMPLE_RATE}s, Press Ctrl+C to stop)")
        try:
            while True:
                chunk = audio_queue.get()   # STEP_SIZE new samples

                # Slide the window forward by one step
                window = np.roll(window, -len(chunk), axis=0)
                window[-len(chunk):] = chunk
                chunks_received += 1

                # Don't analyse until the buffer is fully populated
                if chunks_received < chunks_to_fill:
                    continue

                result = whisper.transcribe_audio_chunk(window)
                if not result:
                    continue

                print(f"Transcription: {result.pharse}")
                detections = detector.detect_keyword(result.pharse)

                now = time.monotonic()
                for d in detections.keywords:
                    if now - last_detection[d.pharse] >= DETECTION_COOLDOWN:
                        print(f"  >> KEYWORD '{d.pharse}' detected (confidence: {d.confidence:.2f})")
                        last_detection[d.pharse] = now

        except KeyboardInterrupt:
            print("\nStopping transcription pipeline...")

if __name__ == "__main__":
    main()
