import torch
from diffusers import DiffusionPipeline
import gradio as gr
from PIL import Image

import sys
sys.path.append("./")

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# Load the Riffusion model pipeline
model_id = "riffusion/riffusion-model-v1"
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_music(prompt: str, num_inference_steps: int, width: int, height: int):
    # Generate music using the Riffusion model with a specific number of inference steps and image size
    output = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width)

    # Saving the generated spectrogram image
    output.images[0].save("generated_music.png")  # Use PNG to preserve metadata

    # Load the spectrogram image
    image = Image.open("generated_music.png")

    # Get spectrogram parameters from the image's EXIF data
    params = SpectrogramParams()

    # Create the SpectrogramImageConverter
    converter = SpectrogramImageConverter(params=params)

    # Convert the image back to audio
    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
        # max_value=None,
    )

    # Export the audio segment to a file
    segment.export("generated_music.wav", format="wav")

    return "generated_music.wav"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Riffusion Music Generator 🎶")
    
    prompt = gr.Textbox(
        label="Enter a music prompt",
        placeholder="E.g., a classical piano piece, upbeat jazz, or ambient soundscape..."
    )
    inference_steps = gr.Slider(
        minimum=10, maximum=100, step=1, value=50, label="Number of Inference Steps"
    )
    width = gr.Slider(
        minimum=128, maximum=2048, step=128, value=512, label="Spectrogram Width (affects duration)"
    )
    height = gr.Slider(
        minimum=128, maximum=2048, step=128, value=512, label="Spectrogram Height (affects duration)"
    )
    
    output_audio = gr.Audio(label="Generated Music")
    
    generate_btn = gr.Button("Generate Music")
    generate_btn.click(fn=generate_music, inputs=[prompt, inference_steps, width, height], outputs=output_audio)

# Launch the app
demo.launch()
