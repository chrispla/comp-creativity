"""Code for music visualizer based on image diffusion using text descriptions
extracted from audio.
"""

import json
import os
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from flamingo.inference.inference import caption_from_file

# load audio
sr = 16000
y, sr = librosa.load("input.mp3", sr=sr)

# split into 10 second chunks
chunk_length = 10 * sr
num_chunks = len(y) // chunk_length
y_chunks = []
for i in range(num_chunks):
    start = i * chunk_length
    end = (i + 1) * chunk_length
    y_chunks.append(y[start:end])
for i, chunk in enumerate(y_chunks):
    if not os.path.exists("./flamingo/files_to_process"):
        os.makedirs("./flamingo/files_to_process")
    sf.write(f"./flamingo/files_to_process/chunk_{i}.wav", chunk, sr)

# create jsonl file for captioning with audio flamingo
caption_prompt = "Generate a descriptive caption for the following audio. Make sure to comment on its vibe/mood as a music piece"
lines = []
for i in range(num_chunks):
    lines.append(
        {
            "path": f"./flamingo/files_to_process/chunk_{i}.wav",
            "prompt": caption_prompt,
        }
    )
# write to jsonl file in ./flamingo/files_to_process/inference.jsonl
with open("./flamingo/files_to_process/inference.jsonl", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(json.dumps(line) + "\n")
# caption
captions = caption_from_file("./flamingo/files_to_process/inference.jsonl")
for i in range(len(captions)):
    if captions[i] == "no response" or captions[i] == "no caption":
        try:
            captions[i] = captions[i-1]
        except:
            try:
                captions[i] = captions[i+1]
            except:
                pass

# delete folder
shutil.rmtree("./flamingo/files_to_process")

# # get chords from the whole track
# ...
# ..

# # get sequences of features over time
# features = []
# for chunk in y_chunks:
#     mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
#     chromagram = librosa.feature.chroma_stft(y=chunk, sr=sr)
#     loudness = librosa.feature.rms(y=chunk, sr=sr) # Simple RMS as perceptual loudness

#     features.append([mfccs, chromagram, loudness])


# # get major turn points in these features
# # categorize them as "major increase", "medium increase", "major decrease", "medium decrease"
# # and keep their timestamps
# def encode_changes(features):
#     """
#     Encodes changes in MFCCs, Chromagram, and Loudness using a checkerboard kernel.
#     """
#     changes = []
#     for mfccs, chromagram, loudness in features:
#         # Calculate change in each feature
#         mfcc_change = np.abs(mfccs[:, -1] - mfccs[:, 0]) #Difference between last and first
#         chroma_change = np.abs(chromagram[:, -1] - chromagram[:, 0])
#         loudness_change = np.abs(loudness[-1] - loudness[0])

#         changes.append([mfcc_change, chroma_change, loudness_change])
#     return changes

# def categorize_changes(changes):
#     """
#     Categorizes changes in MFFCs, Chromagram, and Loudness based on thresholds.
#     Uses pandas for efficient timestamp management.
#     """
#     df = pd.DataFrame(changes, columns=['mfcc_change', 'chroma_change', 'loudness_change'])
#     df['timestamp'] = df.index  # or use a real timestamp if you have it

#     # Define thresholds (tune these!)
#     major_threshold = 0.2
#     medium_threshold = 0.1
#     minor_threshold = 0.05  # Not currently used, but good for future extension

#     # Categorization based on thresholds
#     df['category'] = ''
#     df.loc[df['mfcc_change'] > major_threshold, 'category'] = 'Major Increase'
#     df.loc[(df['mfcc_change'] > medium_threshold) & (df['mfcc_change'] <= major_threshold),
# 'category'] = 'Medium Increase'
#     df.loc[(df['chroma_change'] > major_threshold), 'category'] = 'Major Decrease'
#     df.loc[(df['chroma_change'] > medium_threshold) & (df['chroma_change'] <= major_threshold),
# 'category'] = 'Medium Decrease'
#     df.loc[(df['loudness_change'] > major_threshold), 'category'] = 'Major Decrease'
#     df.loc[(df['loudness_change'] > medium_threshold) & (df['loudness_change'] <=
# major_threshold), 'category'] = 'Medium Decrease'

#     return df

# changes = encode_changes(features)
# categorized_data = categorize_changes(changes)


# # generate prompt for each timestamp that exists, adding the right chord and feature stuff
# ...
# ...
