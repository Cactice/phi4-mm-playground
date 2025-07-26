#!/usr/bin/env python3
"""
Phi-4 Multimodal ONNX Inference Pipeline

Pure functional implementation for running Phi-4 multimodal models with ONNX Runtime.
Supports audio transcription with text prompts.

Key Features:
- Uses standard ONNX Runtime only (no genai dependency)
- Supports audio transcription conditioned on text prompts
- Pure functional programming (no classes)
- Compatible with Phi-4-Multimodal-ONNX-INT4-CPU models
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer
import torchaudio
import torch
import soundfile as sf


# Audio configuration constants
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80

# Token constants
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
AUDIO_PLACEHOLDER = "<|audio_1|>"
AUDIO_SPECIAL_TOKEN = "<|endoftext11|>"
AUDIO_SPECIAL_TOKEN_ID = 200011
END_TOKEN_ID = 200020
EOS_TOKEN_ID = 199999


# Audio processing functions
def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return waveform and sample rate."""
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)  # Convert to mono
    return waveform, sr


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return waveform

    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform_tensor).squeeze().numpy()


def extract_log_mel_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract log-Mel spectrogram features from audio waveform."""
    if sr != SAMPLE_RATE:
        waveform = resample_audio(waveform, sr, SAMPLE_RATE)

    waveform_tensor = torch.from_numpy(waveform).float()
    if len(waveform_tensor.shape) == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window_fn=torch.hamming_window,
    )

    mel_spec = mel_transform(waveform_tensor)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    log_mel_features = log_mel.squeeze(0).transpose(0, 1).numpy()

    return log_mel_features


def prepare_audio_inputs(log_mel_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Prepare audio inputs for the speech ONNX model."""
    num_frames, n_mels = log_mel_features.shape
    assert n_mels == 80, f"Expected 80 Mel bins, got {n_mels}"

    audio_embeds = log_mel_features[np.newaxis, :, :].astype(np.float32)
    audio_attention_mask = np.ones((1, num_frames), dtype=bool)
    audio_sizes = np.array([num_frames], dtype=np.int64)
    audio_projection_mode = np.array([2], dtype=np.int64)

    return {
        "audio_embeds": audio_embeds,
        "audio_attention_mask": audio_attention_mask,
        "audio_sizes": audio_sizes,
        "audio_projection_mode": audio_projection_mode,
    }


# Tokenizer functions
def load_tokenizer(model_path: str) -> GPT2Tokenizer:
    """Load and configure the tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))

    special_tokens = {
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }

    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            setattr(tokenizer, token_type, token)

    return tokenizer


def format_prompt(user_prompt: str, include_audio: bool = True) -> str:
    """Format prompt in Phi-4 chat format with optional audio placeholder."""
    if include_audio:
        audio_placeholder = AUDIO_SPECIAL_TOKEN * 166
        prompt_with_audio = f"{AUDIO_PLACEHOLDER}{audio_placeholder}{user_prompt}"
        formatted_prompt = (
            f"{USER_TOKEN}{prompt_with_audio}{END_TOKEN}{ASSISTANT_TOKEN}"
        )
    else:
        formatted_prompt = f"{USER_TOKEN}{user_prompt}{END_TOKEN}{ASSISTANT_TOKEN}"

    return formatted_prompt


def encode_prompt(
    tokenizer: GPT2Tokenizer, prompt: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode prompt and replace audio placeholders with special tokens."""
    processed_prompt = prompt.replace(AUDIO_PLACEHOLDER, AUDIO_SPECIAL_TOKEN)

    encoded = tokenizer(
        processed_prompt,
        return_tensors="np",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    return input_ids, attention_mask


def decode_tokens(
    tokenizer: GPT2Tokenizer, token_ids: np.ndarray, skip_special_tokens: bool = True
) -> str:
    """Decode token IDs back to text."""
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()

    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]  # Remove batch dimension

    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return decoded_text.strip()


# Model configuration functions
def load_model_config(model_path: Path) -> Dict:
    """Load model configuration from genai_config.json."""
    config_path = model_path / "genai_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("model", {}).get("text", {})
    else:
        return {
            "num_hidden_layers": 32,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "hidden_size": 3072,
        }


def load_lora_adapter(adapter_path: Path) -> Optional[ort.LoraAdapter]:
    """Load a LoRA adapter from file."""
    if not adapter_path.exists():
        return None

    adapter = ort.LoraAdapter()
    adapter.Load(str(adapter_path))
    return adapter


def create_run_options_with_adapters(adapters: List[ort.LoraAdapter]) -> ort.RunOptions:
    """Create RunOptions with active adapters."""
    run_options = ort.RunOptions()
    for adapter in adapters:
        if adapter is not None:
            run_options.add_active_adapter(adapter)
    return run_options


def load_onnx_models(model_path: Path, providers: List[str] = None):
    """Load all ONNX model sessions."""
    providers = providers or ["CPUExecutionProvider"]

    speech_model_path = model_path / "phi-4-mm-speech.onnx"
    speech_session = ort.InferenceSession(str(speech_model_path), providers=providers)

    speech_adapter_path = model_path / "phi-4-mm-speech.onnx_adapter"
    speech_adapter = load_lora_adapter(speech_adapter_path)

    embedding_model_path = model_path / "phi-4-mm-embedding.onnx"
    embedding_session = ort.InferenceSession(
        str(embedding_model_path), providers=providers
    )

    text_model_path = model_path / "phi-4-mm-text.onnx"
    text_session = ort.InferenceSession(str(text_model_path), providers=providers)

    return speech_session, speech_adapter, embedding_session, text_session


def run_speech_model(
    speech_session, speech_adapter, audio_inputs: Dict[str, np.ndarray]
) -> np.ndarray:
    """Run the speech encoder model."""
    # Temporarily disable adapter usage due to compatibility issues
    outputs = speech_session.run(None, audio_inputs)
    audio_features = outputs[0]
    return audio_features


def run_embedding_model(
    embedding_session, input_ids: np.ndarray, audio_features: np.ndarray
) -> np.ndarray:
    """Run the embedding model to merge text and audio embeddings."""
    embed_inputs = {
        "input_ids": input_ids.astype(np.int64),
        "image_features": np.zeros((0, 3072), dtype=np.float32),
        "audio_features": audio_features.astype(np.float32),
    }

    outputs = embedding_session.run(None, embed_inputs)
    inputs_embeds = outputs[0]
    return inputs_embeds


def run_text_model(
    text_session,
    model_config: Dict,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    run_options: Optional[ort.RunOptions],
    past_key_values: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run the text (LLM) model for generation."""
    text_inputs = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
    }

    if past_key_values is not None:
        for i, kv in enumerate(past_key_values):
            if i % 2 == 0:
                text_inputs[f"past_key_values.{i // 2}.key"] = kv
            else:
                text_inputs[f"past_key_values.{i // 2}.value"] = kv
    else:
        num_layers = model_config.get("num_hidden_layers", 32)
        num_kv_heads = model_config.get("num_key_value_heads", 8)
        hidden_size = model_config.get("hidden_size", 3072)
        num_attention_heads = model_config.get("num_attention_heads", 24)
        head_dim = hidden_size // num_attention_heads

        for layer_idx in range(num_layers):
            text_inputs[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (1, num_kv_heads, 0, head_dim), dtype=np.float32
            )
            text_inputs[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (1, num_kv_heads, 0, head_dim), dtype=np.float32
            )

    outputs = text_session.run(None, text_inputs, run_options)
    logits = outputs[0]
    new_past_key_values = []
    for i in range(1, len(outputs)):
        new_past_key_values.append(outputs[i])

    return logits, new_past_key_values


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax of input array."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def generate_text(
    text_session,
    embedding_session,
    model_config: Dict,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    run_options: ort.RunOptions,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[int]:
    """Generate text using iterative decoding."""
    generated_tokens = []
    past_key_values = None
    current_attention_mask = attention_mask.copy()
    current_inputs_embeds = inputs_embeds.copy()

    logits, past_key_values = run_text_model(
        text_session,
        model_config,
        current_inputs_embeds,
        current_attention_mask,
        past_key_values,
    )

    for step in range(max_new_tokens):
        next_token_logits = logits[0, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        if top_p < 1.0:
            sorted_indices = np.argsort(next_token_logits)[::-1]
            sorted_logits = next_token_logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))

            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.any():
                first_removed = np.where(sorted_indices_to_remove)[0][0]
                sorted_indices_to_remove[:first_removed] = False
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float(
                    "-inf"
                )

        next_token = np.argmax(next_token_logits)

        generated_tokens.append(int(next_token))

        if next_token == EOS_TOKEN_ID or next_token == END_TOKEN_ID:
            break

        # For subsequent iterations, we need to convert the new token to embeddings
        # For subsequent tokens, use empty audio features since audio is only needed for the initial prompt
        new_token_ids = np.array([[next_token]], dtype=np.int64)

        # Use empty audio features for subsequent tokens
        empty_audio_features = np.zeros((0, 3072), dtype=np.float32)
        new_token_embeds = run_embedding_model(
            embedding_session, new_token_ids, empty_audio_features
        )

        # Concatenate new token embeddings with previous embeddings
        current_inputs_embeds = np.concatenate(
            [current_inputs_embeds, new_token_embeds], axis=1
        )

        # Run text model with the concatenated embeddings
        logits, past_key_values = run_text_model(
            text_session,
            model_config,
            current_inputs_embeds,
            current_attention_mask[:, -1:],
            run_options,
            past_key_values,
        )

    return generated_tokens


def transcribe_audio(
    model_path: str,
    audio_path: str,
    user_prompt: str = "Transcribe the audio clip into text.",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Main function to transcribe audio with a text prompt."""
    model_path = Path(model_path)

    # Load models and configuration
    model_config = load_model_config(model_path)
    speech_session, speech_adapter, embedding_session, text_session = load_onnx_models(
        model_path
    )
    run_options = create_run_options_with_adapters([speech_adapter])
    tokenizer = load_tokenizer(model_path)

    # Process audio
    waveform, sr = load_audio(audio_path)
    log_mel_features = extract_log_mel_features(waveform, sr)
    audio_inputs = prepare_audio_inputs(log_mel_features)

    # Extract audio features
    audio_features = run_speech_model(speech_session, speech_adapter, audio_inputs)

    # Prepare text prompt
    formatted_prompt = format_prompt(user_prompt, include_audio=True)
    input_ids, attention_mask = encode_prompt(tokenizer, formatted_prompt)

    # Merge embeddings
    inputs_embeds = run_embedding_model(embedding_session, input_ids, audio_features)

    # Generate text
    generated_token_ids = generate_text(
        text_session,
        embedding_session,
        model_config,
        inputs_embeds,
        attention_mask,
        run_options,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Decode result
    transcription = decode_tokens(
        tokenizer, generated_token_ids, skip_special_tokens=True
    )
    return transcription


def main():
    """Example usage of the Phi-4 multimodal ONNX pipeline."""
    model_path = "/Users/yuya/git/phi4-mm-playground-2/Phi-4-Multimodal-ONNX-INT4-CPU"
    sample_audio_path = "/Users/yuya/git/phi4-mm-playground-2/1272-141231-0002.mp3"

    if not Path(model_path).exists():
        print(f"Model path does not exist: {model_path}")
        return

    if Path(sample_audio_path).exists():
        result = transcribe_audio(
            model_path=model_path,
            audio_path=sample_audio_path,
            user_prompt="Transcribe the audio clip into text.",
            max_new_tokens=20,
            temperature=1.0,
        )

        print(f"\n{'=' * 60}")
        print("TRANSCRIPTION RESULT")
        print(f"{'=' * 60}")
        print(f"Audio file: {sample_audio_path}")
        print(f"Transcription: {result}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
