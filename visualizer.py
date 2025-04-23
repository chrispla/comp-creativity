"""Code for music visualizer based on image diffusion using text descriptions
extracted from audio.
"""

import json
import os
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from flamingo.inference.inference import caption_from_file

# load audio
sr = 16000
y, sr = librosa.load("input.mp3", sr=sr)

# # split into 10 second chunks
# chunk_length = 10 * sr
# num_chunks = len(y) // chunk_length
# y_chunks = []
# for i in range(num_chunks):
#     start = i * chunk_length
#     end = (i + 1) * chunk_length
#     y_chunks.append(y[start:end])
# for i, chunk in enumerate(y_chunks):
#     if not os.path.exists("./flamingo/files_to_process"):
#         os.makedirs("./flamingo/files_to_process")
#     sf.write(f"./flamingo/files_to_process/chunk_{i}.wav", chunk, sr)

# # create jsonl file for captioning with audio flamingo
# caption_prompt = "Generate a descriptive caption for the following audio. Make sure to comment on its vibe/mood as a music piece"
# lines = []
# for i in range(num_chunks):
#     lines.append(
#         {
#             "path": f"./flamingo/files_to_process/chunk_{i}.wav",
#             "prompt": caption_prompt,
#         }
#     )
# # write to jsonl file in ./flamingo/files_to_process/inference.jsonl
# with open("./flamingo/files_to_process/inference.jsonl", "w", encoding="utf-8") as f:
#     for line in lines:
#         f.write(json.dumps(line) + "\n")
# # caption
# captions = caption_from_file("./flamingo/files_to_process/inference.jsonl")
# for i in range(len(captions)):
#     if captions[i] == "no response" or captions[i] == "no caption":
#         try:
#             captions[i] = captions[i - 1]
#         except:
#             try:
#                 captions[i] = captions[i + 1]
#             except:
#                 pass

# # delete folder
# shutil.rmtree("./flamingo/files_to_process")

# # get chords from the whole track
# ...

# get sequences of features over time
hop_length = 512 * 8  # Use consistent hop length
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8, hop_length=hop_length)
loudness = librosa.feature.rms(
    y=y, frame_length=2048 * 8, hop_length=hop_length
)  # Simple RMS as perceptual loudness

# plot each feature
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.title("MFCCs")
plt.imshow(mfccs, aspect="auto", origin="lower")
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title("Loudness")
plt.plot(loudness[0])
plt.savefig("features.png")


# get major turn points in these features
def detect_significant_changes(mfccs, loudness, window_size=4):
    """
    Detects significant changes in MFCCs and loudness over time using a sliding window.
    Returns two separate dataframes for MFCC and loudness changes.
    """
    # Initialize lists for MFCC changes
    mfcc_timestamps = []
    mfcc_changes = []
    mfcc_directions = []

    # Initialize lists for loudness changes
    loudness_timestamps = []
    loudness_changes = []
    loudness_directions = []

    # Define thresholds
    major_mfcc_threshold = 0.5
    medium_mfcc_threshold = 0.2
    major_loudness_threshold = 0.02
    medium_loudness_threshold = 0.01

    # Use a sliding window to detect changes
    for i in range(window_size, mfccs.shape[1]):
        # Calculate change in MFCCs
        mfcc_diff = np.mean(np.abs(mfccs[:, i] - mfccs[:, i - window_size]))
        mfcc_direction = (
            "Increase"
            if np.mean(mfccs[:, i]) > np.mean(mfccs[:, i - window_size])
            else "Decrease"
        )
        # Track MFCC changes separately
        if mfcc_diff > medium_mfcc_threshold:
            mfcc_timestamps.append(i)
            mfcc_changes.append(mfcc_diff)
            category = "Major" if mfcc_diff > major_mfcc_threshold else "Medium"
            mfcc_directions.append(f"{category} {mfcc_direction}")

    for i in range(window_size, loudness.shape[1]):
        # Calculate change in loudness
        loudness_diff = np.abs(loudness[0, i] - loudness[0, i - window_size])
        loudness_direction = (
            "Increase" if loudness[0, i] > loudness[0, i - window_size] else "Decrease"
        )

        # Track loudness changes separately
        if loudness_diff > medium_loudness_threshold:
            loudness_timestamps.append(i)
            loudness_changes.append(loudness_diff)
            category = "Major" if loudness_diff > major_loudness_threshold else "Medium"
            loudness_directions.append(f"{category} {loudness_direction}")

    # Create DataFrame for MFCC changes
    mfcc_df = pd.DataFrame(
        {
            "frame": mfcc_timestamps,
            "time_sec": [
                frame_to_time(t, sr, hop_length=hop_length) for t in mfcc_timestamps
            ],
            "mfcc_change": mfcc_changes,
            "category": mfcc_directions,
        }
    )

    # Create DataFrame for loudness changes
    loudness_df = pd.DataFrame(
        {
            "frame": loudness_timestamps,
            "time_sec": [
                frame_to_time(t, sr, hop_length=hop_length) for t in loudness_timestamps
            ],
            "loudness_change": loudness_changes,
            "category": loudness_directions,
        }
    )

    return mfcc_df, loudness_df


def frame_to_time(frame_number, sr, hop_length=512):
    """Convert frame number to time in seconds."""
    return frame_number * hop_length / sr


# Usage:
mfcc_changes, loudness_changes = detect_significant_changes(mfccs, loudness)
print("MFCC Changes:")
print(mfcc_changes.head(5))
print("\nLoudness Changes:")
print(loudness_changes.head(5))

# Plot with separate change points
plt.figure(figsize=(12, 8))

# Plot MFCCs with MFCC change points
plt.subplot(2, 1, 1)
plt.title("MFCCs with change points")
plt.imshow(mfccs, aspect="auto", origin="lower")
plt.colorbar()
for idx, row in mfcc_changes.iterrows():
    plt.axvline(x=row["frame"], color="r", linestyle="--", alpha=0.5)

# Plot Loudness with loudness change points
plt.subplot(2, 1, 2)
plt.title("Loudness with change points")
plt.plot(loudness[0])
for idx, row in loudness_changes.iterrows():
    plt.axvline(x=row["frame"], color="g", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("features_with_changes.png")
