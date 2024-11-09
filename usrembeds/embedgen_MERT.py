from datautils.dataset import MusicDataset
import torch
import json
import os
import time
from transformers import AutoModel, Wav2Vec2FeatureExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    music_path = "D:/music"
    stats_path = "embeddings/clean_stats.csv"
    emb_path = "embeddings"

    SAVE_RATE = 500  # Save every 500 tracks
    BATCH_SIZE = 32

    # Load your dataset here
    dataset = MusicDataset(
        music_path,
        stats_path,
        resample=24000,  # MERT-v1-95M expects 24kHz audio
        repeat=1,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Load MERT-v1-95M model and feature extractor
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    emb_dict = {"metadata": {"model": "MERT-v1-95M"}}
    part = 0
    mean_time = 0
    start_time = time.time()
    for idx, track in enumerate(dataloader):
        stat, audio = track
        audio = audio.to(DEVICE)

        # Process each audio in the batch
        for i in range(audio.size(0)):
            input_audio = audio[i].cpu().numpy()
            inputs = feature_extractor(input_audio, sampling_rate=24000, return_tensors="pt", padding=True)
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states).squeeze()
                mean_emb = hidden_states.mean(dim=(0, 1)).cpu().numpy()

            track_id = stat["id"][i]
            if track_id not in emb_dict:
                emb_dict[track_id] = []
            emb_dict[track_id].append(mean_emb.tolist())

        if (idx+1) % SAVE_RATE == 0:
            with open(os.path.join(emb_path, f"embeddings_part_{part}.json"), "w") as f:
                json.dump(emb_dict, f)
            part += 1
            emb_dict = {}

        end_time = time.time()
        mean_time = (mean_time * idx + (end_time - start_time)) / (idx + 1)
        print(f"Processed {idx + 1}/{len(dataloader)} batches in {(end_time - start_time):.2f} seconds (mean: {mean_time:.2f})")
        remaining_time = (len(dataloader) - idx - 1) * mean_time
        print(f"Estimated time remaining: {remaining_time/60/60:.0f}h {remaining_time/60%60:.0f}m {remaining_time%60:.0f}s")
        start_time = time.time()

    # Save any remaining embeddings
    if emb_dict:
        with open(os.path.join(emb_path, f"embeddings_part_{part}.json"), "w") as f:
            json.dump(emb_dict, f)
