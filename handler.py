"""
MT3 RunPod Serverless Worker
Multi-Task Multitrack Music Transcription using Google Magenta's MT3 model

Supports:
- Multi-instrument transcription (piano, strings, winds, brass, percussion, etc.)
- Up to 128 MIDI programs (General MIDI)
- Polyphonic audio processing
- Orchestra-grade transcription

Based on: https://github.com/magenta/mt3
Paper: https://arxiv.org/abs/2111.03017
"""

import runpod
import requests
import tempfile
import os
import functools
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v2 as tf
tf.config.set_visible_devices([], 'GPU')  # Use JAX for GPU, not TF

import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

# Check JAX device
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Sample rate for MT3 models
SAMPLE_RATE = 16000

# Model paths
MODEL_DIR = "/models"
SF2_PATH = "/models/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2"


class InferenceModel:
    """
    MT3 Inference wrapper for T5X-based music transcription.
    Adapted from the official Colab notebook.
    """

    def __init__(self, checkpoint_path: str, model_type: str = "mt3"):
        """
        Initialize the MT3 inference model.

        Args:
            checkpoint_path: Path to the model checkpoint directory
            model_type: 'mt3' for multi-instrument or 'ismir2021' for piano-only
        """
        # Model-specific configurations
        if model_type == "ismir2021":
            self.num_velocity_bins = 127
            self.inputs_length = 512
            gin_files = [
                os.path.join(checkpoint_path, "gin/model.gin"),
                os.path.join(checkpoint_path, "gin/inference.gin"),
            ]
        else:  # mt3
            self.num_velocity_bins = 1
            self.inputs_length = 256
            gin_files = [
                os.path.join(checkpoint_path, "gin/model.gin"),
            ]

        # Parse gin configuration
        gin.clear_config()
        gin.parse_config_files_and_bindings(
            gin_files,
            bindings=[
                f"preprocessors.compute_spectrograms.spectrogram_config = @spectrograms.SpectrogramConfig()",
                "spectrograms.SpectrogramConfig.overlap = 0.5",
                "spectrograms.SpectrogramConfig.hop_width = 128",
                "spectrograms.SpectrogramConfig.num_mel_bins = 512",
            ],
            finalize_config=False,
        )

        # Build codec and vocabulary
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=self.num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)

        # Build output features for seqio
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        # Build encoding spec
        self.encoding_spec = note_sequences.NoteEncodingSpec(
            codec=self.codec
        )

        # Load model config
        model_config = models.T5MusicTransformerConfig(
            vocab_size=self.vocabulary.vocab_size,
            dtype='float32',
            emb_dim=512,
            num_heads=8,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=2048,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False,
        )

        # Build T5X model
        module = network.MusicTransformer(config=model_config)

        self.model = models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.vocabulary,
            output_vocabulary=self.vocabulary,
            optimizer_def=None,
            input_depth=spectrograms.input_depth(
                spectrograms.SpectrogramConfig()
            ),
        )

        # Restore checkpoint
        self.restore_from_checkpoint(checkpoint_path)

    def restore_from_checkpoint(self, checkpoint_path: str):
        """Restore model weights from checkpoint."""
        # Get checkpoint state
        checkpointer = t5x.checkpoints.Checkpointer(
            train_state=None,
            partitioner=t5x.partitioning.PjitPartitioner(num_partitions=1),
            checkpoints_dir=checkpoint_path,
        )

        # Create dummy input for initialization
        dummy_input = {
            "encoder_input_tokens": jax.numpy.zeros((1, self.inputs_length, 512)),
            "decoder_input_tokens": jax.numpy.zeros((1, 1024), dtype=jax.numpy.int32),
        }

        # Initialize model params
        self.params = self.model.get_initial_variables(
            jax.random.PRNGKey(0),
            dummy_input,
        )["params"]

        # Restore from checkpoint
        state_dict = checkpointer.restore(path=checkpoint_path)
        self.params = state_dict["target"]

        # JIT compile predict function
        @functools.partial(jax.jit, static_argnums=(0,))
        def predict_batch(model, params, batch):
            return model.predict_batch(params, batch)

        self._predict_fn = functools.partial(predict_batch, self.model)

    def _audio_to_frames(self, audio: np.ndarray) -> tuple:
        """Convert audio to spectrogram frames."""
        spectrogram_config = spectrograms.SpectrogramConfig()

        # Compute spectrogram
        frames, frame_times = spectrograms.split_audio(
            audio, spectrogram_config
        )

        return frames, frame_times

    def __call__(self, audio: np.ndarray) -> note_seq.NoteSequence:
        """
        Transcribe audio to note sequence.

        Args:
            audio: Audio samples at 16kHz sample rate

        Returns:
            note_seq.NoteSequence with transcribed notes
        """
        # Convert to spectrogram frames
        frames, frame_times = self._audio_to_frames(audio)

        # Process in chunks
        predictions = []
        num_frames = len(frames)

        for i in range(0, num_frames, self.inputs_length):
            # Get chunk
            chunk_frames = frames[i:i + self.inputs_length]
            chunk_times = frame_times[i:i + self.inputs_length]

            # Pad if needed
            if len(chunk_frames) < self.inputs_length:
                pad_len = self.inputs_length - len(chunk_frames)
                chunk_frames = np.pad(
                    chunk_frames,
                    ((0, pad_len), (0, 0)),
                    mode='constant'
                )

            # Create batch
            batch = {
                "encoder_input_tokens": chunk_frames[np.newaxis, ...],
                "decoder_input_tokens": np.zeros((1, 1024), dtype=np.int32),
            }

            # Run inference
            result = self._predict_fn(self.params, batch)
            predictions.append({
                "est_tokens": result[0],
                "start_time": chunk_times[0] if len(chunk_times) > 0 else i * 0.032,
            })

        # Convert predictions to note sequence
        result = metrics_utils.event_predictions_to_ns(
            predictions,
            codec=self.codec,
            encoding_spec=self.encoding_spec,
        )

        return result["est_ns"]


# Pre-load models at startup
print("Loading MT3 models...")
MODELS = {}

# Load MT3 (multi-instrument) model
mt3_path = os.path.join(MODEL_DIR, "mt3")
if os.path.exists(mt3_path):
    try:
        MODELS["mt3"] = InferenceModel(mt3_path, model_type="mt3")
        print("MT3 model loaded successfully!")
    except Exception as e:
        print(f"Failed to load MT3 model: {e}")

# Load ISMIR2021 (piano) model
ismir_path = os.path.join(MODEL_DIR, "ismir2021")
if os.path.exists(ismir_path):
    try:
        MODELS["ismir2021"] = InferenceModel(ismir_path, model_type="ismir2021")
        print("ISMIR2021 model loaded successfully!")
    except Exception as e:
        print(f"Failed to load ISMIR2021 model: {e}")

print(f"Loaded models: {list(MODELS.keys())}")


# MIDI program to instrument name mapping (General MIDI)
MIDI_PROGRAM_NAMES = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet", 8: "Celesta", 9: "Glockenspiel",
    10: "Music Box", 11: "Vibraphone", 12: "Marimba", 13: "Xylophone",
    14: "Tubular Bells", 15: "Dulcimer", 16: "Drawbar Organ", 17: "Percussive Organ",
    18: "Rock Organ", 19: "Church Organ", 20: "Reed Organ", 21: "Accordion",
    22: "Harmonica", 23: "Tango Accordion", 24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)",
    29: "Overdriven Guitar", 30: "Distortion Guitar", 31: "Guitar Harmonics",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    35: "Fretless Bass", 36: "Slap Bass 1", 37: "Slap Bass 2",
    38: "Synth Bass 1", 39: "Synth Bass 2", 40: "Violin", 41: "Viola",
    42: "Cello", 43: "Contrabass", 44: "Tremolo Strings", 45: "Pizzicato Strings",
    46: "Orchestral Harp", 47: "Timpani", 48: "String Ensemble 1",
    49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs", 54: "Synth Choir", 55: "Orchestra Hit",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)", 84: "Lead 5 (charang)", 85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)", 88: "Pad 1 (new age)",
    89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)",
    95: "Pad 8 (sweep)", 96: "FX 1 (rain)", 97: "FX 2 (soundtrack)",
    98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)", 100: "FX 5 (brightness)",
    101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto",
    108: "Kalimba", 109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock",
    116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum",
    119: "Reverse Cymbal", 120: "Guitar Fret Noise", 121: "Breath Noise",
    122: "Seashore", 123: "Bird Tweet", 124: "Telephone Ring",
    125: "Helicopter", 126: "Applause", 127: "Gunshot",
}


def download_audio(url: str, output_path: str) -> None:
    """Download audio file with streaming."""
    with requests.get(url, timeout=300, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)


def upload_to_supabase(
    data: bytes,
    bucket: str,
    storage_path: str,
    supabase_url: str,
    service_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload data to Supabase Storage and return public URL."""
    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{storage_path}"

    headers = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    response = requests.post(upload_url, headers=headers, data=data, timeout=120)
    response.raise_for_status()

    return f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"


def handler(event):
    """
    RunPod handler for MT3 multi-instrument transcription.

    Input:
        audio_url: URL to audio file
        model_type: (optional) 'mt3' for multi-instrument or 'ismir2021' for piano
        output_midi: (optional) Whether to return MIDI file URL (default: false)
        storage_bucket: (optional) Supabase bucket for MIDI output
        storage_prefix: (optional) Storage path prefix

    Output:
        notes: List of NoteEvent objects with instrument information
        note_count: Total number of detected notes
        instruments: Dict mapping program numbers to instrument names
        tracks: Notes organized by instrument/track
        midi_url: (optional) URL to generated MIDI file
    """
    audio_path = None

    try:
        input_data = event.get("input", {})
        audio_url = input_data.get("audio_url")

        if not audio_url:
            return {"error": "audio_url is required"}

        model_type = input_data.get("model_type", "mt3")
        output_midi = input_data.get("output_midi", False)
        storage_bucket = input_data.get("storage_bucket", "transcriptions")
        storage_prefix = input_data.get("storage_prefix", "mt3")

        # Check model availability
        if model_type not in MODELS:
            available = list(MODELS.keys())
            return {
                "error": f"Model '{model_type}' not available. Available: {available}"
            }

        model = MODELS[model_type]

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Determine file extension
        ext = ".wav"
        url_lower = audio_url.lower()
        if ".mp3" in url_lower:
            ext = ".mp3"
        elif ".flac" in url_lower:
            ext = ".flac"
        elif ".ogg" in url_lower:
            ext = ".ogg"
        elif ".m4a" in url_lower:
            ext = ".m4a"

        audio_path = os.path.join(temp_dir, f"input{ext}")

        # Download audio
        print(f"Downloading audio from: {audio_url}")
        download_audio(audio_url, audio_path)

        # Load and resample audio to 16kHz mono
        print("Loading and resampling audio...")
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Run transcription
        print(f"Running {model_type} transcription...")
        note_sequence = model(audio)

        # Convert to output format
        notes = []
        instruments = {}
        tracks = {}  # Organize by instrument

        for note in note_sequence.notes:
            program = note.program
            instrument_name = MIDI_PROGRAM_NAMES.get(program, f"Program {program}")

            note_event = {
                "pitch": note.pitch,
                "startTime": float(note.start_time),
                "duration": float(note.end_time - note.start_time),
                "velocity": note.velocity,
                "instrument": program,
                "instrumentName": instrument_name,
            }

            notes.append(note_event)
            instruments[program] = instrument_name

            # Organize by track
            if program not in tracks:
                tracks[program] = {
                    "program": program,
                    "name": instrument_name,
                    "notes": [],
                }
            tracks[program]["notes"].append(note_event)

        # Sort notes by start time
        notes.sort(key=lambda x: x["startTime"])
        for track in tracks.values():
            track["notes"].sort(key=lambda x: x["startTime"])

        result = {
            "notes": notes,
            "note_count": len(notes),
            "instruments": instruments,
            "tracks": list(tracks.values()),
        }

        # Optionally generate and upload MIDI
        if output_midi:
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if supabase_url and supabase_key:
                import uuid

                # Generate MIDI from note sequence
                midi_data = note_seq.note_sequence_to_midi_file(note_sequence)

                # Upload to Supabase
                job_id = str(uuid.uuid4())[:8]
                storage_path = f"{storage_prefix}/{job_id}/transcription.mid"

                midi_url = upload_to_supabase(
                    midi_data,
                    storage_bucket,
                    storage_path,
                    supabase_url,
                    supabase_key,
                    content_type="audio/midi",
                )

                result["midi_url"] = midi_url

        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download audio: {str(e)}"}
    except Exception as e:
        import traceback
        return {
            "error": f"Transcription failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
    finally:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            try:
                import shutil
                shutil.rmtree(os.path.dirname(audio_path))
            except Exception:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
