import os
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from transformers import ClapModel, ClapProcessor
from sklearn.metrics.pairwise import cosine_similarity

AUDIO_DIR = "my_audio"
SAMPLE_RATE = 48000  # CLAP requires 48kHz

GENRES = {
    "Caprice_No24.wav": "classical",
    "Fur_Elise.wav": "classical",
    "Toccata_Fugue.wav":  "classical",
    "Master_of_Puppets.wav": "metal",
    "Cowboys_From_Hell.wav": "metal",
    "Ace_Of_Spades.wav": "metal",
    "MR_RECOUP.wav": "rap",
    "PLEDGE.wav": "rap",
    "Took_Her_To_The_O.wav": "rap",
    "Riptide.wav": "pop"
}

print("Loading CLAP model...")
model_id = "laion/clap-htsat-unfused"

processor = ClapProcessor.from_pretrained(model_id) # Converts sound waves into a format for the model
model = ClapModel.from_pretrained(model_id)         # Returns the audio feature vector
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded. Device: {device}\n")

# Embeddings
def get_audio_embedding(filepath: str):
    # Loads an audio file and returns the embedding as a numpy vector
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    inputs = processor(
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.audio_model(**inputs)

    # L2 Normalization (standard for cosine similarity)
    embedding = output.pooler_output    # Audio file converted to a feature vector
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)    # Vector scaling
    return embedding.cpu().numpy().flatten()


files = list(GENRES.keys())
embeddings = {}

print("Processing audio files:")
for name in files:
    filepath = os.path.join(AUDIO_DIR, name)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    print(f"{name}")
    embeddings[name] = get_audio_embedding(filepath)


# Cosine similarity matrix
names = list(embeddings.keys())
matrix = np.array([embeddings[n] for n in names])
sim_matrix = cosine_similarity(matrix)

df_sim = pd.DataFrame(sim_matrix, index=names, columns=names)

plt.figure(figsize=(10, 8))
sns.heatmap(
    df_sim,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    vmin=0, vmax=1,
    linewidths=0.5,
)
plt.title("Cosine Similarity Matrix (CLAP embeddings)", pad=15)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("similarity_heatmap.png", dpi=150)
plt.show()
print("Heatmap saved as: similarity_heatmap.png")

pairs = []
for a, b in combinations(names, 2):
    score = df_sim.loc[a, b]
    genre_a = GENRES[a]
    genre_b = GENRES[b]
    match = "YES" if genre_a == genre_b else "NO"
    pairs.append({
        "File A": a,
        "File B": b,
        "Genre A": genre_a,
        "Genre B": genre_b,
        "Similarity": round(score, 4),
        "Same Genre": match,
    })

df_pairs = pd.DataFrame(pairs).sort_values("Similarity", ascending=False)

# Save to CSV
df_pairs.to_csv("similarity_results.csv", index=False)
print("Results saved to: similarity_results.csv")