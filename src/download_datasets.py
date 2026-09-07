"""
Automated dataset downloader and unpacker.
Supports:
1. FMA-small metadata & audio (Free Music Archive)
2. MagnaTagATune annotations
3. MusicCaps captions (Google Research)
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import csv
import json


DATASET_URLS = {
    "fma_metadata": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
    "fma_small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "musiccaps_csv": "https://raw.githubusercontent.com/google-research-datasets/musiccaps/main/musiccaps-public.csv",
}


def download_file(url: str, dest_path: str):
    """Downloads a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading from {url} to {dest_path}...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / (total_size + 1e-6))
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print("\nDownload complete.")


def unzip_archive(zip_path: str, extract_to: str):
    """Extracts a ZIP archive."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def setup_musiccaps(data_dir: str = "data/raw"):
    """Downloads MusicCaps annotations and converts them to parsed JSON."""
    csv_path = os.path.join(data_dir, "musiccaps.csv")
    download_file(DATASET_URLS["musiccaps_csv"], csv_path)

    json_path = os.path.join(data_dir, "musiccaps.json")
    if os.path.exists(csv_path) and not os.path.exists(json_path):
        records = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "ytid": row.get("ytid", ""),
                    "start_s": row.get("start_s", 0),
                    "end_s": row.get("end_s", 10),
                    "aspect_list": row.get("aspect_list", ""),
                    "caption": row.get("caption", ""),
                    "is_audioset_eval": row.get("is_audioset_eval", False),
                })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Parsed {len(records)} MusicCaps annotations into {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Download official datasets for GNN-BERT project.")
    parser.add_argument("--dataset", type=str, choices=["fma_small", "fma_metadata", "musiccaps", "all"], default="musiccaps")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()

    if args.dataset in ["musiccaps", "all"]:
        setup_musiccaps(args.output_dir)

    if args.dataset in ["fma_metadata", "all"]:
        zip_path = os.path.join(args.output_dir, "fma_metadata.zip")
        download_file(DATASET_URLS["fma_metadata"], zip_path)
        unzip_archive(zip_path, args.output_dir)

    if args.dataset in ["fma_small", "all"]:
        zip_path = os.path.join(args.output_dir, "fma_small.zip")
        download_file(DATASET_URLS["fma_small"], zip_path)
        unzip_archive(zip_path, args.output_dir)


if __name__ == "__main__":
    main()
