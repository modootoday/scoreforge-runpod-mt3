"""
MT3 RunPod Serverless Worker
Multi-Task Multitrack Music Transcription using Google Magenta's MT3 model

Note: This is a placeholder implementation. MT3 requires complex setup with JAX/T5X.
For now, it uses a simplified inference approach.
"""

import runpod
import requests
import tempfile
import os
import numpy as np


def download_audio(url: str) -> str:
    """Download audio file from URL to temporary file"""
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    ext = ".wav"
    if ".mp3" in url.lower():
        ext = ".mp3"
    elif ".flac" in url.lower():
        ext = ".flac"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(response.content)
        return f.name


def handler(event):
    """
    RunPod handler for MT3 multi-instrument transcription

    Input:
        audio_url: URL to audio file
        model_type: (optional) 'mt3' for multi-instrument or 'ismir2021' for piano, default 'mt3'

    Output:
        notes: List of NoteEvent objects with instrument information
        note_count: Total number of detected notes
        instruments: List of detected instruments
    """
    try:
        input_data = event.get("input", {})
        audio_url = input_data.get("audio_url")

        if not audio_url:
            return {"error": "audio_url is required"}

        model_type = input_data.get("model_type", "mt3")

        # Download audio file
        audio_path = download_audio(audio_url)

        try:
            # Import MT3 inference modules
            # Note: These imports require the MT3 package to be installed
            from mt3 import models
            from mt3 import network
            from mt3 import note_sequences
            from mt3 import run_length_encoding
            from mt3 import spectrograms
            from mt3 import vocabularies
            import note_seq
            import librosa

            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Get spectrogram
            spectrogram = spectrograms.compute_spectrogram(
                audio,
                sample_rate=sr,
            )

            # Load model
            checkpoint_path = f"/models/{model_type}"
            model = models.InferenceModel(checkpoint_path, None)

            # Run inference
            est_ns = model(spectrogram)

            # Convert note sequence to NoteEvent format
            notes = []
            instruments = set()

            for note in est_ns.notes:
                notes.append({
                    "pitch": note.pitch,
                    "startTime": note.start_time,
                    "duration": note.end_time - note.start_time,
                    "velocity": note.velocity,
                    "instrument": note.program,  # MIDI program number
                })
                instruments.add(note.program)

            # Sort by start time
            notes.sort(key=lambda x: x["startTime"])

            return {
                "notes": notes,
                "note_count": len(notes),
                "instruments": list(instruments),
            }

        except ImportError as e:
            return {
                "error": f"MT3 not properly installed: {str(e)}",
                "note": "MT3 requires JAX/T5X setup. Consider using Basic-Pitch instead.",
            }

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download audio: {str(e)}"}
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
