#!/usr/bin/env python3
"""
Download pre-trained models for baby monitor ML features.

This script downloads:
1. MoveNet Lightning (pose estimation) - 4.2 MB
2. (Optional) EfficientDet-Lite (object detection) - 4.4 MB
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "movenet_lightning": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
        "filename": "movenet_lightning.tflite",
        "size": "4.2 MB",
        "description": "Pose estimation for sleeping position detection"
    },
    "movenet_thunder": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite",
        "filename": "movenet_thunder.tflite",
        "size": "12 MB",
        "description": "Higher accuracy pose estimation (slower, optional)"
    }
}


def download_file(url, destination, description):
    """Download file with progress bar"""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {destination}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    print("="*60)
    print("Baby Monitor ML Models Downloader")
    print("="*60)

    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} is installed")
    except ImportError:
        print("⚠ TensorFlow not found. Install with:")
        print("  pip install tensorflow  (desktop)")
        print("  pip install tflite-runtime  (Raspberry Pi)")
        print("\nContinuing with download anyway...")

    print(f"\nModels will be saved to: {MODELS_DIR.absolute()}")

    # Ask which models to download
    print("\nAvailable models:")
    for i, (key, info) in enumerate(MODELS.items(), 1):
        print(f"{i}. {info['filename']} - {info['size']}")
        print(f"   {info['description']}")

    print("\nDownload options:")
    print("1. MoveNet Lightning (recommended, fast)")
    print("2. MoveNet Thunder (higher accuracy, slower)")
    print("3. Both models")

    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"

    models_to_download = []
    if choice == "1":
        models_to_download = ["movenet_lightning"]
    elif choice == "2":
        models_to_download = ["movenet_thunder"]
    elif choice == "3":
        models_to_download = ["movenet_lightning", "movenet_thunder"]
    else:
        print("Invalid choice. Downloading MoveNet Lightning by default.")
        models_to_download = ["movenet_lightning"]

    # Download selected models
    success_count = 0
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        destination = MODELS_DIR / model_info["filename"]

        # Check if already exists
        if destination.exists():
            overwrite = input(f"\n{model_info['filename']} already exists. Overwrite? (y/n) [n]: ").strip().lower()
            if overwrite != 'y':
                print("Skipping...")
                success_count += 1
                continue

        # Download
        if download_file(model_info["url"], destination, model_info["description"]):
            success_count += 1
            print(f"✓ Saved to: {destination}")

    # Summary
    print("\n" + "="*60)
    print(f"Downloaded {success_count}/{len(models_to_download)} models successfully")
    print("="*60)

    if success_count > 0:
        print("\nNext steps:")
        print("1. Test pose estimation: python models/pose_estimator.py")
        print("2. Update config.yaml to enable ML features")
        print("3. Run baby monitor: python main.py")

        # Check model file size
        for model_key in models_to_download:
            model_path = MODELS_DIR / MODELS[model_key]["filename"]
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"\n{model_path.name}: {size_mb:.1f} MB")

    else:
        print("\n⚠ No models were downloaded.")
        print("You can manually download from:")
        for model_key in models_to_download:
            print(f"  {MODELS[model_key]['url']}")


if __name__ == '__main__':
    main()
