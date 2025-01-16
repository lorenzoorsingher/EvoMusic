from setuptools import setup, find_packages

setup(
    name="EvoMusic",
    version="0.1.0",
    description="Evolve your own personalized music",
    # author="Your Name",
    # author_email="your_email@example.com",
    # url="https://your_project_url.com",
    packages=find_packages(),  # Automatically discovers all modules and sub-modules
    install_requires=[
        # Add your dependencies here, e.g., 'numpy', 'requests>=2.25.1'
        'julius==0.2.7',
        'matplotlib==3.9.2',
        'numpy==1.24.3',
        'pandas==2.2.3',
        'pillow==11.0.0',
        'pyparsing==3.2.0',
        'python-dateutil==2.9.0.post0',
        'python-dotenv==1.0.1',
        'PyYAML==6.0.1',
        'requests==2.32.3',
        'resampy==0.2.2',
        'scikit-learn==1.5.2',
        'scipy==1.13.1',
        'sox==1.5.0',
        'ssqueezepy==0.6.5',
        'torch==2.5.0',
        'torchaudio==2.5.0',
        'torchmetrics==1.5.1',
        'torchvision==0.20.0',
        'tqdm==4.66.5',
        # triton==3.1.0
        'typing_extensions==4.12.2',
        'wandb==0.18.5',
        'nnAudio == 0.3.3',
        'diffusers==0.32.2',
        'transformers==4.47.1',
        'pydub==0.25.1',
        'evotorch==0.5.1',
        'PySoundFile==0.9.0.post1',
        'gradio==3.36.1',
    ],
    python_requires='>=3.9',  # Specify the Python version requirement
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
