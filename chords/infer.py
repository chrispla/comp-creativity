import argparse
import os
import warnings

import mir_eval
import pretty_midi as pm

from .btc_model import *
from .utils import logger
from .utils.mir_eval_modules import (
    audio_file_to_features,
    get_audio_paths,
    idx2chord,
    idx2voca_chord,
)

warnings.filterwarnings("ignore")
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def predict_chords(filepath, voca=False, save_dir="./test"):
    """
    Predict chords from an audio file and return the chord annotations.

    Args:
        filepath (str): Path to the audio file (.mp3 or .wav)
        voca (bool): Whether to use large vocabulary chord recognition
        save_dir (str): Directory to save the .lab and .midi files

    Returns:
        list: List of tuples containing (start_time, end_time, chord)
    """
    config = HParams.load("run_config.yaml")

    # Initialize mean and std with default values
    mean = 0
    std = 1

    if voca:
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170
        model_file = "./chords/test/btc_model_large_voca.pt"
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        model_file = "./chords/test/btc_model.pt"
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")

    model = BTC_model(config=config.model).to(device)

    # Load model
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        mean = checkpoint["mean"]
        std = checkpoint["std"]
        model.load_state_dict(checkpoint["model"])
        logger.info("restore model")
    else:
        logger.warning(
            f"Model file {model_file} not found. Using default normalization values."
        )

    logger.info("Processing file: %s" % filepath)
    # Load audio file
    feature, feature_per_second, song_length_second = audio_file_to_features(
        filepath, config
    )
    logger.info("audio file loaded and feature computation success : %s" % filepath)

    # Chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model["timestep"]

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(
        feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0
    )
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    chord_annotations = []

    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(
                feature[:, n_timestep * t : n_timestep * (t + 1), :]
            )
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    end_time = time_unit * (n_timestep * t + i)
                    lines.append(
                        "%.3f %.3f %s\n"
                        % (start_time, end_time, idx_to_chord[prev_chord])
                    )
                    chord_annotations.append(
                        (start_time, end_time, idx_to_chord[prev_chord])
                    )
                    start_time = end_time
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        end_time = time_unit * (n_timestep * t + i)
                        lines.append(
                            "%.3f %.3f %s\n"
                            % (start_time, end_time, idx_to_chord[prev_chord])
                        )
                        chord_annotations.append(
                            (start_time, end_time, idx_to_chord[prev_chord])
                        )
                    break

    return chord_annotations


# Main execution when script is called directly
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voca", default=True, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument("--audio_dir", type=str, default="./test")
    parser.add_argument("--save_dir", type=str, default="./test")
    args = parser.parse_args()

    # Process all audio files in the directory
    audio_paths = get_audio_paths(args.audio_dir)

    for i, audio_path in enumerate(audio_paths):
        logger.info(
            "======== %d of %d in progress ========" % (i + 1, len(audio_paths))
        )
        predict_chords(audio_path, args.voca, args.save_dir)
