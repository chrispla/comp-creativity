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
# ..

# get sequences of features over time
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
loudness = librosa.feature.rms(
    y=y, frame_length=2048 * 8, hop_length=512 * 8
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
# categorize them as "major increase", "medium increase", "major decrease", "medium decrease"
# and keep their timestamps
def detect_significant_changes(mfccs, loudness, window_size=20):
    """
    Detects significant changes in MFCCs and loudness over time using a sliding window.
    Returns a dataframe with timestamps of significant changes.

    Args:
        mfccs: MFCC features with shape (n_mfcc, n_frames)
        loudness: RMS loudness with shape (1, n_frames)
        window_size: Size of window to detect changes (in frames)
    """
    # Number of frames (use the minimum to handle different lengths)
    n_frames = min(mfccs.shape[1], loudness.shape[1])

    # Initialize lists to store results
    timestamps = []
    mfcc_changes = []
    loudness_changes = []
    categories = []

    # Define thresholds
    major_threshold = 0.5
    medium_threshold = 0.2

    # Use a sliding window to detect changes
    for i in range(window_size, n_frames):
        # Calculate change in MFCCs (average across all coefficients)
        mfcc_diff = np.mean(np.abs(mfccs[:, i] - mfccs[:, i - window_size]))

        # Calculate change in loudness
        loudness_diff = np.abs(loudness[0, i] - loudness[0, i - window_size])

        # Only add points with significant changes
        if mfcc_diff > medium_threshold or loudness_diff > medium_threshold:
            timestamps.append(i)
            mfcc_changes.append(mfcc_diff)
            loudness_changes.append(loudness_diff)

            # Categorize the change
            if mfcc_diff > major_threshold:
                categories.append("Major MFCC Change")
            elif mfcc_diff > medium_threshold:
                categories.append("Medium MFCC Change")
            elif loudness_diff > major_threshold:
                categories.append("Major Loudness Change")
            elif loudness_diff > medium_threshold:
                categories.append("Medium Loudness Change")
            else:
                categories.append("Minor Change")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "frame": timestamps,  # Keep the frame number
            "time_sec": [
                frame_to_time(t, sr, hop_length=512 * 8) for t in timestamps
            ],  # Convert frames to seconds
            "mfcc_change": mfcc_changes,
            "loudness_change": loudness_changes,
            "category": categories,
        }
    )

    return df


def frame_to_time(frame_number, sr, hop_length=512):
    """Convert frame number to time in seconds."""
    return frame_number * hop_length / sr


# Replace your encode_changes and categorize_changes calls with:
significant_changes = detect_significant_changes(mfccs, loudness)
print(significant_changes)

# Optional: Plot the significant change points on your features
plt.figure(figsize=(12, 8))

# Plot MFCCs
plt.subplot(2, 1, 1)
plt.title("MFCCs with change points")
plt.imshow(mfccs, aspect="auto", origin="lower")
plt.colorbar()
# Mark significant changes
for idx, row in significant_changes.iterrows():
    if "MFCC" in row["category"]:
        plt.axvline(x=row["frame"], color="r", linestyle="--", alpha=0.5)

# Plot Loudness
plt.subplot(2, 1, 2)
plt.title("Loudness with change points")
plt.plot(loudness[0])
# Mark significant changes
for idx, row in significant_changes.iterrows():
    if "Loudness" in row["category"]:
        plt.axvline(x=row["frame"], color="r", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("features_with_changes.png")
