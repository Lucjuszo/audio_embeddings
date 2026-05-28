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

## 📑 Szczegółowe wyniki podobieństwa (CSV)
Poniżej znajduje się pełne zestawienie wszystkich 45 przebadanych par ułożonych od najwyższego podobieństwa. Zauważ, że model świetnie radzi sobie z grupowaniem utworów tego samego gatunku (wartość `TAK`).

<details>
<summary><b>Rozwiń pełną tabelę wyników (46 wierszy)</b></summary>

| Plik A | Plik B | Gatunek A | Gatunek B | Podobieństwo | Ten sam gatunek |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Took_Her_To_The_O.wav | PLEDGE.wav | rap | rap | 0.9412 | TAK |
| Master_of_Puppets.wav | Cowboys_From_Hell.wav | metal | metal | 0.9105 | TAK |
| Caprice_No24.wav | Fur_Elise.wav | classical | classical | 0.8873 | TAK |
| MR_RECOUP.wav | PLEDGE.wav | rap | rap | 0.8521 | TAK |
| Ace_Of_Spades.wav | Master_of_Puppets.wav | metal | metal | 0.8314 | TAK |
| ... | ... | ... | ... | ... | ... |
| Fur_Elise.wav | Took_Her_To_The_O.wav | classical | rap | 0.1124 | NIE |
| Caprice_No24.wav | Cowboys_From_Hell.wav | classical | metal | 0.0841 | NIE |

*(Pełne dane znajdziesz w pliku `similarity_results.csv` w repozytorium).*
</details>

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
   **3. Stwórz folder o nazwie moje_audio w głównym katalogu projektu.
4. Wrzuć do niego własne pliki .wav.
5. Dostosuj słownik GATUNKI w pliku audio_ai.py do swoich plików.
6. Uruchom skrypt: python audio_ai.py.**
