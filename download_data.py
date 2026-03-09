"""
Script to download Twitter Sentiment Analysis (Hate Speech) dataset from Kaggle.
Requires: pip install kaggle
Setup: Place kaggle.json (API credentials) in ~/.kaggle/ or %USERPROFILE%\.kaggle\
"""
import os
import zipfile
from pathlib import Path

def download_dataset():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        data_dir = Path(__file__).parent / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_files(
            "arkhoshghalb/twitter-sentiment-analysis-hatred-speech",
            path=str(data_dir)
        )
        
        # Extract if zip was downloaded
        for zip_path in data_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            zip_path.unlink()
            print(f"Extracted {zip_path.name}")
        
        print("Dataset downloaded successfully to data/raw/")
    except ImportError:
        print("Install kaggle: pip install kaggle")
        print("Then download manually from:")
        print("https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech")
    except Exception as e:
        print(f"Error: {e}")
        print("Manual download: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech")
        print("Place train.csv in data/raw/")

if __name__ == "__main__":
    download_dataset()
