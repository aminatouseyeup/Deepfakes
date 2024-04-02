# Initializing all the encoder libraries
from IPython.display import Audio
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


encoder_weights = Path(
    "deepfake-audio-generator/Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt"
)
vocoder_weights = Path(
    "deepfake-audio-generator/Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt"
)
syn_dir = Path(
    "deepfake-audio-generator/Real-Time-Voice-Cloning/synthesizer/saved_models/logs-pretrained/taco_pretrained"
)
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


def generate_audio(text, input_path):

    in_fpath = Path(input_path)
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    with io.capture_output() as captured:
        specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    sf.write("output.wav", generated_wav, synthesizer.sample_rate)
