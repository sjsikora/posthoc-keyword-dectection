from dataclasses import dataclass
from abc import abstractmethod
from config import SAMPLE_RATE
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

@dataclass
class TranscriptionOutput():
    pharse: str
    confidence : float | None

class STTModel():
    @abstractmethod
    def transcribe_audio_chunk(self, audio_chunk) -> TranscriptionOutput | None:
        pass
class WhisperModel(STTModel):

    LOG_PROB_THRESHOLD = -1

    def __init__(self) -> None:
        print("Loading Hugging Face Whisper model (this may take a moment)...")
        model_id = "openai/whisper-base"

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        print(f"Model loaded successfully on {self.device}!")

    def transcribe_audio_chunk(self, audio_chunk) -> TranscriptionOutput | None:
        audio_data = audio_chunk.flatten()

        input_features = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(self.device)

        # 1. Ask the model to return its internal scoring data
        outputs = self.model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True
        )

        predicted_ids = outputs.sequences

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()

        if not transcription:
            return None

        # 2. Extract the log probabilities of the generated tokens
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # 3. Convert log probabilities to standard probabilities (0.0 to 1.0)
        # Using [0] because we are processing a batch size of 1
        token_probabilities = transition_scores[0]

        # 4. Calculate the overall sequence confidence
        # Taking the mean of all token probabilities is the standard approach
        sequence_confidence = token_probabilities.mean().item()

        if sequence_confidence < self.LOG_PROB_THRESHOLD:
            return None

        return TranscriptionOutput(transcription, sequence_confidence)
