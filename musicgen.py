import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import gradio as gr
import numpy as np

# Suppress warnings (optional, but can be helpful during development)
import warnings
warnings.filterwarnings("ignore")

# Load the MusicGen model and processor with the 'eager' attention implementation
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    # attn_implementation="eager"  # Addresses Flash Attention and empty attention mask warnings
)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Set the device to GPU if available, else CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate music
def generate_music(text_prompt, guidance_scale, max_new_tokens):
    try:
        # Preprocess the input text
        inputs = processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate audio using the model
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens
        )

        # Retrieve the sampling rate from the model's configuration
        sampling_rate = model.config.audio_encoder.sampling_rate

        # Convert the generated audio to a NumPy array
        audio = audio_values[0].cpu().numpy()

        # Handle multi-channel audio (e.g., stereo)
        if audio.ndim == 2 and audio.shape[0] in [1, 2]:
            # Transpose to shape (samples, channels) for Gradio
            audio = audio.T

        # Ensure the audio data is in float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize the audio to ensure values are within [-1.0, 1.0]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Clip the audio to prevent any out-of-range values
        audio = np.clip(audio, -1.0, 1.0)

        # Check for non-finite values and handle them
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains non-finite values (NaN or Infinity).")

        return (sampling_rate, audio)

    except Exception as e:
        print(f"Error in generate_music: {e}")
        # Return silence or an empty array in case of error
        silent_audio = np.zeros(1, dtype=np.float32)
        return (sampling_rate, silent_audio)

# Create the Gradio interface using updated components
iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(lines=2, label="Text Prompt"),
        gr.Slider(1, 5, step=0.5, value=3, label="Guidance Scale"),
        gr.Slider(256, 1500, step=256, value=1024, label="Max New Tokens")
    ],
    outputs=gr.Audio(type="numpy", label="Generated Music"),
    title="MusicGen Text-to-Music Generator",
    description="Enter a text prompt to generate music using MusicGen.",
)

# Launch the interface with share=True for a public link
iface.launch(share=True)
