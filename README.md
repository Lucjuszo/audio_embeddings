# audio_embeddings
Measure the mathematical acoustic similarity between different audio files.

# 🎵 Audio Similarity AI

Proste, ale potężne narzędzie do analizy i porównywania utworów muzycznych przy użyciu sztucznej inteligencji. Projekt wykorzystuje zaawansowany model **CLAP (Contrastive Language-Audio Pretraining)**, aby ocenić matematyczne podobieństwo brzmieniowe między różnymi plikami audio i sprawdzić, jak AI grupuje utwory w ramach gatunków muzycznych.

## 🚀 Jak to działa?
1. Skrypt wczytuje pliki `.wav` z określonego folderu.
2. Model CLAP analizuje każdy utwór i przekształca go na **embedding** (wielowymiarowy wektor cech dźwięku).
3. Obliczane jest **podobieństwo cosinusowe** (cosine similarity) między wektorami każdego z utworów.
4. Wyniki są eksportowane do pliku `.csv` oraz wizualizowane w postaci mapy ciepła (heatmap).

## 📊 Wyniki działania (Heatmapa)
Poniższa macierz przedstawia wyniki analizy. Im cieplejszy kolor i wynik bliższy `1.0`, tym większe podobieństwo między utworami zdaniem sztucznej inteligencji:

![Macierz podobieństwa cosinusowego](similarity_heatmap.png)

## 🛠️ Technologie
* **Python 3**
* **PyTorch** & **Transformers (Hugging Face)** – ładowanie i obsługa modelu LAION CLAP.
* **Librosa** – przetwarzanie sygnału audio (skalowanie do 48kHz, konwersja na mono).
* **Scikit-learn** – obliczanie podobieństwa cosinusowego.
* **Pandas, Matplotlib, Seaborn** – analiza danych i generowanie wykresów.

## ⚙️ Jak uruchomić lokalnie?
Ze względów prawnych repozytorium nie zawiera plików audio. Aby przetestować kod, utwórz folder `moje_audio` i umieść w nim własne pliki w formacie `.wav`. Następnie dostosuj słownik `GATUNKI` w kodzie do swoich plików.

1. Sklonuj repozytorium.
2. Zainstaluj wymagane biblioteki:

   ```bash
   pip install torch librosa numpy pandas matplotlib seaborn transformers scikit-learn
3. Stwórz folder o nazwie moje_audio w głównym katalogu projektu.
4. Wrzuć do niego własne pliki .wav.
5. Dostosuj słownik GATUNKI w pliku audio_ai.py do swoich plików.
6. Uruchom skrypt: python audio_ai.py.

   
