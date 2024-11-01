import gradio as gr
from diffusers import MusicLDMPipeline
import torch
import scipy

# Load the pipeline
repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Function to generate music and save as .wav
def generate_music(prompt, num_inference_steps, audio_length_in_s):
    audio = pipe(prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s).audios[0]
    
    # Save the audio file
    scipy.io.wavfile.write("output.wav", rate=16000, data=audio)
    
    # Return the file path to play in Gradio
    return "output.wav"

# Define the Gradio interface
interface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter music prompt here"),
        gr.Slider(label="Number of Inference Steps", minimum=1, maximum=1000, step=1, value=10),
        gr.Slider(label="Audio Length (seconds)", minimum=1.0, maximum=1000.0, step=0.5, value=5.0)
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="Music Generator with MusicLDM",
    description="Generate music based on prompts using the MusicLDM model."
)

# Launch the interface
interface.launch()
