# Copyright (c) 2025 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import snapshot_download
from pydub import AudioSegment
from safetensors.torch import load_file
from tqdm import tqdm

from flamingo.inference.src.factory import create_model_and_transforms
from flamingo.inference.utils import Dict2Class, get_autocast, get_cast_dtype


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_num_windows(T, sr, clap_config):
    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (
            max_num_window * window_length - (max_num_window - 1) * window_overlap
        )
    else:
        num_windows = 1 + int(
            np.ceil((T - window_length) / float(window_length - window_overlap))
        )
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap

    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000 : (start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)

        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))

        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(
                    data.flatten(), orig_sr=original_sr, target_sr=target_sr
                )
            else:
                data = librosa.resample(
                    data.T, orig_sr=original_sr, target_sr=target_sr
                )[0]
        else:
            if channels != 1:
                data = data.T[0]

    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))

    assert len(data.shape) == 1, data.shape
    return data


def load_audio(audio_path, clap_config):
    sr = 16000
    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = (
        max_num_window * (clap_config["window_length"] - clap_config["window_overlap"])
        + clap_config["window_overlap"]
    )

    audio_data = read_audio(
        audio_path, sr, duration, 0.0, clap_config
    )  # hard code audio start to 0.0
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start : start + window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) < max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)

    return audio_clips, audio_embed_mask


def load_audio_from_data(audio_data, clap_config, sr=16000):
    """
    Load audio from preloaded data instead of from a file path.

    Args:
        audio_data: Numpy array containing preloaded 16kHz audio data
        clap_config: Configuration dictionary for audio processing
        sr: Sample rate (default: 16000)

    Returns:
        Tuple of (audio_clips, audio_embed_mask)
    """
    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    # Ensure the audio data is correctly formatted
    if audio_data.min() >= 0:
        audio_data = 2 * audio_data / abs(audio_data.max()) - 1.0
    else:
        audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start : start + window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) < max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)

    return audio_clips, audio_embed_mask


class AudioFlamingoProcessor:
    """
    A class to load the Audio Flamingo model once and process multiple audio files efficiently.
    """

    def __init__(self, model_path="./", token=None, device_id=0, batch_size=1):
        """
        Initialize the Audio Flamingo model and tokenizer.

        Args:
            model_path: Path to save or load model files
            token: Optional Hugging Face token for private models
            device_id: GPU device ID to use
            batch_size: Number of audio files to process in a single batch
        """
        self.model_path = model_path
        self.device_id = device_id
        self.batch_size = batch_size

        # Download model if needed
        if token:
            snapshot_download(
                repo_id="nvidia/audio-flamingo-2-0.5B",
                local_dir=model_path,
                token=token,
            )
        else:
            snapshot_download(
                repo_id="nvidia/audio-flamingo-2-0.5B", local_dir=model_path
            )

        # Load configuration
        config = yaml.load(
            open("./flamingo/inference/configs/inference.yaml"),
            Loader=yaml.FullLoader,
        )
        self.model_config = config["model_config"]
        self.clap_config = config["clap_config"]
        args = Dict2Class(config["train_config"])

        # Initialize model and tokenizer
        self.model, self.tokenizer = create_model_and_transforms(
            **self.model_config,
            clap_config=self.clap_config,
            use_local_files=args.offline,
            gradient_checkpointing=args.gradient_checkpointing,
            freeze_lm_embeddings=args.freeze_lm_embeddings,
        )

        self.model = self.model.to(device_id)
        self.model.eval()

        # Load weights
        with open(f"{model_path}/safe_ckpt/metadata.json", "r") as f:
            metadata = json.load(f)

        state_dict = {}
        for chunk_name in metadata:
            chunk_path = f"{model_path}/safe_ckpt/{chunk_name}.safetensors"
            chunk_tensors = load_file(chunk_path)
            state_dict.update(chunk_tensors)

        self.model.load_state_dict(state_dict, False)

        # Set up inference parameters
        self.autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))
        self.cast_dtype = get_cast_dtype(args.precision)
        self.inference_kwargs = {
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.95,
            "num_return_sequences": 1,
        }

    def predict_from_audio_data(self, audio_data_list, questions):
        """
        Generate captions for a list of audio data arrays.

        Args:
            audio_data_list: List of numpy arrays containing audio data (16kHz mono)
            questions: List of questions/prompts corresponding to each audio file

        Returns:
            List of generated captions/responses
        """
        results = []

        # Process audio files in batches to manage memory usage
        for i in tqdm(
            range(0, len(audio_data_list), self.batch_size),
            desc="Processing audio batches",
        ):
            batch_audio = audio_data_list[i : i + self.batch_size]
            batch_questions = questions[i : i + self.batch_size]

            batch_results = self._process_batch(batch_audio, batch_questions)
            results.extend(batch_results)

        return results

    def _process_batch(self, batch_audio, batch_questions):
        """Process a batch of audio files"""
        batch_results = []

        for i, audio_data in enumerate(batch_audio):
            question = batch_questions[i]

            # Process audio data
            audio_clips, audio_embed_mask = load_audio_from_data(
                audio_data, self.clap_config
            )
            audio_clips = audio_clips.to(
                self.device_id, dtype=self.cast_dtype, non_blocking=True
            )
            audio_embed_mask = audio_embed_mask.to(
                self.device_id, dtype=self.cast_dtype, non_blocking=True
            )

            # Process text prompt
            text_prompt = str(question).lower()
            sample = f"<audio>{text_prompt.strip()}{self.tokenizer.sep_token}"
            text = self.tokenizer(
                sample,
                max_length=512,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )

            input_ids = text["input_ids"].to(self.device_id, non_blocking=True)
            attention_mask = text["attention_mask"].to(
                self.device_id, non_blocking=True
            )

            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    audio_x=audio_clips.unsqueeze(0),
                    audio_x_mask=audio_embed_mask.unsqueeze(0),
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=256,
                    **self.inference_kwargs,
                )[0]

            output_decoded = (
                self.tokenizer.decode(output)
                .split(self.tokenizer.sep_token)[-1]
                .replace(self.tokenizer.eos_token, "")
                .replace(self.tokenizer.pad_token, "")
                .replace("<|endofchunk|>", "")
            )

            batch_results.append(output_decoded)

        return batch_results

    def predict_from_files(self, file_paths, questions):
        """
        Generate captions for a list of audio file paths.

        Args:
            file_paths: List of paths to audio files
            questions: List of questions/prompts corresponding to each audio file

        Returns:
            List of generated captions/responses
        """
        results = []

        # Process audio files in batches
        for i in tqdm(
            range(0, len(file_paths), self.batch_size), desc="Processing audio files"
        ):
            batch_paths = file_paths[i : i + self.batch_size]
            batch_questions = questions[i : i + self.batch_size]

            batch_results = []
            for j, filepath in enumerate(batch_paths):
                audio_clips, audio_embed_mask = load_audio(filepath, self.clap_config)
                audio_clips = audio_clips.to(
                    self.device_id, dtype=self.cast_dtype, non_blocking=True
                )
                audio_embed_mask = audio_embed_mask.to(
                    self.device_id, dtype=self.cast_dtype, non_blocking=True
                )

                text_prompt = str(batch_questions[j]).lower()
                sample = f"<audio>{text_prompt.strip()}{self.tokenizer.sep_token}"

                text = self.tokenizer(
                    sample,
                    max_length=512,
                    padding="longest",
                    truncation="only_first",
                    return_tensors="pt",
                )

                input_ids = text["input_ids"].to(self.device_id, non_blocking=True)
                attention_mask = text["attention_mask"].to(
                    self.device_id, non_blocking=True
                )

                with torch.no_grad():
                    output = self.model.generate(
                        audio_x=audio_clips.unsqueeze(0),
                        audio_x_mask=audio_embed_mask.unsqueeze(0),
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_new_tokens=256,
                        **self.inference_kwargs,
                    )[0]

                output_decoded = (
                    self.tokenizer.decode(output)
                    .split(self.tokenizer.sep_token)[-1]
                    .replace(self.tokenizer.eos_token, "")
                    .replace(self.tokenizer.pad_token, "")
                    .replace("<|endofchunk|>", "")
                )

                batch_results.append(output_decoded)

            results.extend(batch_results)

        return results


if __name__ == "__main__":
    # Helper function to load audio files
    def load_audio_files(file_paths, sr=16000):
        """Load multiple audio files as numpy arrays"""
        audio_data_list = []
        for path in file_paths:
            audio, _ = librosa.load(path, sr=sr)
            audio_data_list.append(audio)
        return audio_data_list

    # Initialize the processor
    processor = AudioFlamingoProcessor(batch_size=1)

    # Process pre-loaded audio data
    file_paths = [
        "/Users/chris/Desktop/distributed-transcription/audio_0.wav",
        "/Users/chris/Desktop/distributed-transcription/audio_1.wav",
        "/Users/chris/Desktop/distributed-transcription/audio_2.wav",
    ]
    audio_data_list = load_audio_files(file_paths)
    questions = [
        "What sounds do you hear in this audio?",
        "Describe the environment in this recording",
        "What instruments are playing in this clip?",
    ]

    # Generate captions from pre-loaded audio data
    results = processor.predict_from_audio_data(audio_data_list, questions)

    # Print results
    for i, result in enumerate(results):
        print(f"Audio {i + 1}: {result}")
