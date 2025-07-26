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
AUDIO_COMPRESSION_RATE = 8

# STFT Normal (16kHz) configuration
STFT_NORMAL_N_FFT = 512
STFT_NORMAL_FRAME_LENGTH = 400
STFT_NORMAL_HOP_LENGTH = 160
STFT_NORMAL_WIN_FN = "hamming"

# LogMel (16kHz) configuration
LOGMEL_CHUNK_SIZE = 30
LOGMEL_HOP_LENGTH = 160
LOGMEL_N_FFT = 512
LOGMEL_N_MEL = 80
LOGMEL_FEATURE_FIRST = 0
LOGMEL_NO_PADDING = 1

# STFT Normal 8kHz configuration
STFT_NORMAL_8K_N_FFT = 256
STFT_NORMAL_8K_FRAME_LENGTH = 200
STFT_NORMAL_8K_HOP_LENGTH = 80
STFT_NORMAL_8K_WIN_FN = "hamming"

# LogMel 8kHz configuration
LOGMEL_8K_CHUNK_SIZE = 30
LOGMEL_8K_HOP_LENGTH = 80
LOGMEL_8K_N_FFT = 512
LOGMEL_8K_N_MEL = 80
LOGMEL_8K_FEATURE_FIRST = 0
LOGMEL_8K_NO_PADDING = 1

# Legacy constants for backward compatibility
N_FFT = LOGMEL_N_FFT
HOP_LENGTH = LOGMEL_HOP_LENGTH
N_MELS = LOGMEL_N_MEL

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
def get_audio_config() -> Dict[str, any]:
    """Get complete audio processing configuration."""
    return {
        "audio_compression_rate": AUDIO_COMPRESSION_RATE,
        "stft_normal/n_fft": STFT_NORMAL_N_FFT,
        "stft_normal/frame_length": STFT_NORMAL_FRAME_LENGTH,
        "stft_normal/hop_length": STFT_NORMAL_HOP_LENGTH,
        "stft_normal/win_fn": STFT_NORMAL_WIN_FN,
        "logmel/chunk_size": LOGMEL_CHUNK_SIZE,
        "logmel/hop_length": LOGMEL_HOP_LENGTH,
        "logmel/n_fft": LOGMEL_N_FFT,
        "logmel/n_mel": LOGMEL_N_MEL,
        "logmel/feature_first": LOGMEL_FEATURE_FIRST,
        "logmel/no_padding": LOGMEL_NO_PADDING,
        "stft_normal_8k/n_fft": STFT_NORMAL_8K_N_FFT,
        "stft_normal_8k/frame_length": STFT_NORMAL_8K_FRAME_LENGTH,
        "stft_normal_8k/hop_length": STFT_NORMAL_8K_HOP_LENGTH,
        "stft_normal_8k/win_fn": STFT_NORMAL_8K_WIN_FN,
        "logmel_8k/chunk_size": LOGMEL_8K_CHUNK_SIZE,
        "logmel_8k/hop_length": LOGMEL_8K_HOP_LENGTH,
        "logmel_8k/n_fft": LOGMEL_8K_N_FFT,
        "logmel_8k/n_mel": LOGMEL_8K_N_MEL,
        "logmel_8k/feature_first": LOGMEL_8K_FEATURE_FIRST,
        "logmel_8k/no_padding": LOGMEL_8K_NO_PADDING,
    }


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

    # Ensure float32 dtype and proper tensor format for torchaudio
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
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
        n_fft=LOGMEL_N_FFT,
        hop_length=LOGMEL_HOP_LENGTH,
        n_mels=LOGMEL_N_MEL,
        window_fn=torch.hamming_window,
        win_length=STFT_NORMAL_FRAME_LENGTH,
    )

    mel_spec = mel_transform(waveform_tensor)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    log_mel_features = log_mel.squeeze(0).transpose(0, 1).numpy()

    return log_mel_features


def prepare_audio_inputs(log_mel_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Prepare audio inputs for the speech ONNX model."""
    num_frames, n_mels = log_mel_features.shape
    assert n_mels == LOGMEL_N_MEL, f"Expected {LOGMEL_N_MEL} Mel bins, got {n_mels}"

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


def format_prompt(user_prompt: str, audio_size: int) -> str:
    """Format prompt in Phi-4 chat format with audio placeholder and user task prompt."""
    audio_placeholder = AUDIO_SPECIAL_TOKEN * audio_size
    # Follow the recommended format: <|user|><audio>{task prompt}<|end|><|assistant|>{label}<|end|>
    formatted_prompt = (
        f"{USER_TOKEN}{audio_placeholder}{user_prompt}{END_TOKEN}{ASSISTANT_TOKEN}"
    )

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


def run_speech_model(speech_session, audio_inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """Run the speech encoder model."""
    outputs = speech_session.run(None, audio_inputs)
    audio_features = outputs[0]
    return audio_features


def run_embedding_model(
    embedding_session, input_ids: np.ndarray, audio_features: np.ndarray
) -> np.ndarray:
    """Run the embedding model to merge text and audio embeddings."""
    print(
        f"[DEBUG] run_embedding_model called with input_ids.shape={input_ids.shape}, audio_features.shape={audio_features.shape}"
    )

    embed_inputs = {
        "input_ids": input_ids.astype(np.int64),
        "image_features": np.zeros((0, 3072), dtype=np.float32),
        "audio_features": audio_features.astype(np.float32),
    }
    print(f"[DEBUG] Embedding inputs prepared: {list(embed_inputs.keys())}")

    outputs = embedding_session.run(None, embed_inputs)
    inputs_embeds = outputs[0]
    print(f"[DEBUG] Embedding model outputs shape: {inputs_embeds.shape}")
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
    print(
        f"[DEBUG] run_text_model called with inputs_embeds.shape={inputs_embeds.shape}, attention_mask.shape={attention_mask.shape}"
    )
    print(f"[DEBUG] past_key_values provided: {past_key_values is not None}")

    text_inputs = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
    }

    if past_key_values is not None:
        print(
            f"[DEBUG] Using existing past_key_values with {len(past_key_values)} items"
        )
        for i, kv in enumerate(past_key_values):
            if i % 2 == 0:
                text_inputs[f"past_key_values.{i // 2}.key"] = kv
            else:
                text_inputs[f"past_key_values.{i // 2}.value"] = kv
    else:
        print("[DEBUG] Initializing empty past_key_values...")
        num_layers = model_config.get("num_hidden_layers", 32)
        num_kv_heads = model_config.get("num_key_value_heads", 8)
        hidden_size = model_config.get("hidden_size", 3072)
        num_attention_heads = model_config.get("num_attention_heads", 24)
        head_dim = hidden_size // num_attention_heads
        print(
            f"[DEBUG] Model config: layers={num_layers}, kv_heads={num_kv_heads}, hidden={hidden_size}, heads={num_attention_heads}, head_dim={head_dim}"
        )

        for layer_idx in range(num_layers):
            text_inputs[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (1, num_kv_heads, 0, head_dim), dtype=np.float32
            )
            text_inputs[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (1, num_kv_heads, 0, head_dim), dtype=np.float32
            )

    print(f"[DEBUG] Running text session with {len(text_inputs)} inputs...")
    outputs = text_session.run(None, text_inputs, run_options)
    print(f"[DEBUG] Text session returned {len(outputs)} outputs")

    logits = outputs[0]
    print(f"[DEBUG] Logits shape: {logits.shape}")

    new_past_key_values = []
    for i in range(1, len(outputs)):
        new_past_key_values.append(outputs[i])
    print(f"[DEBUG] Created {len(new_past_key_values)} new past_key_values")

    return logits, new_past_key_values


def generate_text(
    text_session,
    embedding_session,
    model_config: Dict,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    run_options: ort.RunOptions,
    max_new_tokens: int = 100,
) -> List[int]:
    """Generate text using iterative decoding with proper KV-cache optimization."""
    print(
        f"[DEBUG] Starting KV-cache optimized text generation with max_new_tokens={max_new_tokens}"
    )
    print(
        f"[DEBUG] Initial inputs_embeds.shape={inputs_embeds.shape}, attention_mask.shape={attention_mask.shape}"
    )

    generated_tokens = []
    past_key_values = None
    full_attention_mask = attention_mask.copy()

    # Initialize current embeddings and run text model for first time
    current_embeds = inputs_embeds

    for step in range(max_new_tokens):
        # Run the text model
        logits, past_key_values = run_text_model(
            text_session,
            model_config,
            current_embeds,
            full_attention_mask,
            run_options,
            past_key_values,
        )
        print(f"[DEBUG] Logits shape: {logits.shape}")

        # Get the next token from the last position of the logits
        next_token_logits = logits[0, -1, :]
        print(
            f"[DEBUG] Next token logits shape: {next_token_logits.shape}, min/max: {np.min(next_token_logits):.3f}/{np.max(next_token_logits):.3f}"
        )

        next_token = np.argmax(next_token_logits)
        print(f"[DEBUG] Selected token: {next_token}")
        generated_tokens.append(int(next_token))

        if next_token == EOS_TOKEN_ID or next_token == END_TOKEN_ID:
            print(f"[DEBUG] Hit end token ({next_token}), stopping generation")
            break

        # Prepare inputs for the next iteration
        new_token_ids = np.array([[next_token]], dtype=np.int64)
        print(f"[DEBUG] New token IDs shape: {new_token_ids.shape}")

        # Get embeddings for only the new token
        empty_audio_features = np.zeros((0, 3072), dtype=np.float32)
        print("[DEBUG] Running embedding model for new token...")
        current_embeds = run_embedding_model(
            embedding_session, new_token_ids, empty_audio_features
        )
        print(f"[DEBUG] New token embeds shape: {current_embeds.shape}")

        # Update the full attention mask to include new token
        new_attention = np.ones((1, 1), dtype=np.int64)
        full_attention_mask = np.concatenate(
            [full_attention_mask, new_attention], axis=1
        )
        print(f"[DEBUG] Updated full attention mask shape: {full_attention_mask.shape}")

    return generated_tokens


def transcribe_audio(
    model_path: str,
    audio_path: str,
    user_prompt: str = "Transcribe the audio clip into text.",
    max_new_tokens: int = 100,
) -> str:
    """Main function to transcribe audio with a text prompt."""
    print("[DEBUG] Starting transcription with parameters:")
    print(f"  - model_path: {model_path}")
    print(f"  - audio_path: {audio_path}")
    print(f"  - user_prompt: {user_prompt}")
    print(f"  - max_new_tokens: {max_new_tokens}")

    model_path = Path(model_path)

    # Load models and configuration
    print("[DEBUG] Loading model configuration...")
    model_config = load_model_config(model_path)
    print(f"[DEBUG] Model config: {model_config}")

    print("[DEBUG] Loading ONNX models...")
    speech_session, speech_adapter, embedding_session, text_session = load_onnx_models(
        model_path
    )
    print(f"[DEBUG] Speech adapter loaded: {speech_adapter is not None}")

    print("[DEBUG] Creating run options with adapters...")
    run_options = create_run_options_with_adapters([speech_adapter])
    print(f"[DEBUG] Run options created: {run_options}")

    print("[DEBUG] Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    print(f"[DEBUG] Tokenizer vocab size: {tokenizer.vocab_size}")

    # Process audio
    print("[DEBUG] Processing audio...")
    waveform, sr = load_audio(audio_path)
    print(f"[DEBUG] Audio loaded: shape={waveform.shape}, sr={sr}")

    log_mel_features = extract_log_mel_features(waveform, sr)
    print(f"[DEBUG] Log-mel features extracted: shape={log_mel_features.shape}")

    audio_inputs = prepare_audio_inputs(log_mel_features)
    print(f"[DEBUG] Audio inputs prepared: {list(audio_inputs.keys())}")

    # Extract audio features
    print("[DEBUG] Running speech model...")
    audio_features = run_speech_model(speech_session, audio_inputs)
    print(f"[DEBUG] Audio features extracted: shape={audio_features.shape}")

    # Prepare text prompt
    print("[DEBUG] Preparing text prompt...")
    audio_size = audio_inputs["audio_sizes"][0] // AUDIO_COMPRESSION_RATE
    formatted_prompt = format_prompt(user_prompt, audio_size)
    print(f"[DEBUG] Formatted prompt: {formatted_prompt[:100]}...")

    input_ids, attention_mask = encode_prompt(tokenizer, formatted_prompt)
    print(
        f"[DEBUG] Encoded prompt: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}"
    )

    # Merge embeddings
    print("[DEBUG] Running embedding model...")
    inputs_embeds = run_embedding_model(embedding_session, input_ids, audio_features)
    print(f"[DEBUG] Inputs embeddings created: shape={inputs_embeds.shape}")

    # Generate text
    print("[DEBUG] Starting text generation...")
    generated_token_ids = generate_text(
        text_session,
        embedding_session,
        model_config,
        inputs_embeds,
        attention_mask,
        run_options,
        max_new_tokens=max_new_tokens,
    )
    print(f"[DEBUG] Generated {len(generated_token_ids)} tokens: {generated_token_ids}")

    # Decode result
    print("[DEBUG] Decoding tokens...")
    transcription = decode_tokens(
        tokenizer, generated_token_ids, skip_special_tokens=True
    )
    print(f"[DEBUG] Final transcription: {transcription}")
    return transcription


def main():
    """Example usage of the Phi-4 multimodal ONNX pipeline."""
    model_path = "/Users/yuya/git/phi4-mm-playground-2/Phi-4-Multimodal-ONNX-INT4-CPU"
    sample_audio_path = "/Users/yuya/git/phi4-mm-playground-2/record.mp3"

    if not Path(model_path).exists():
        print(f"Model path does not exist: {model_path}")
        return

    if Path(sample_audio_path).exists():
        result = transcribe_audio(
            model_path=model_path,
            audio_path=sample_audio_path,
            user_prompt="Transcribe the audio clip into text.",
            max_new_tokens=40,
        )

        print(f"\n{'=' * 60}")
        print("TRANSCRIPTION RESULT")
        print(f"{'=' * 60}")
        print(f"Audio file: {sample_audio_path}")
        print(f"Transcription: {result}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
