"""
Download the official Roboflow example COCO-format segmentation dataset.

Supports two modes:
1. Authenticated download via roboflow SDK (requires API key in config.yaml).
2. Fallback to pure Python requests for public datasets (no API key).
"""

import os
import zipfile
from urllib.parse import urlencode

import requests
from tqdm import tqdm

from utils import load_config, resolve_dataset_path


def get_dataset_download_url(workspace: str, project: str, version: int) -> str:
    """Build a direct download URL for a public Roboflow dataset."""
    return (
        f"https://universe.roboflow.com/ds/{workspace}/{project}/{version}"
        f"?{urlencode({'format': 'coco', 'key': 'public'})}"
    )


def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file to the specified directory."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted {zip_path} -> {extract_to}")


def download_with_roboflow_sdk(api_key: str, workspace: str, project: str, version: int, location: str):
    """Download dataset using the authenticated roboflow SDK."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError(
            "roboflow package is required for authenticated download. "
            "Install it with: pip install roboflow"
        ) from exc

    print(f"Downloading dataset via Roboflow SDK (workspace={workspace}, project={project}, version={version}) ...")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download("coco", location=location)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def main():
    cfg = load_config("config.yaml")
    dataset_dir = resolve_dataset_path(cfg)

    # Check if dataset already exists and looks valid
    required_splits = ["train", "valid", "test"]
    if all(
        os.path.isdir(os.path.join(dataset_dir, sp)) for sp in required_splits
    ):
        print(f"Dataset already exists at {dataset_dir}, skipping download.")
        return

    example = cfg.get("example_dataset", {})
    workspace = "roboflow-jvuqo"
    project = "creacks-eapny"
    version = example.get("version", 4)
    api_key = cfg.get("roboflow_api_key", "")

    os.makedirs(os.path.dirname(dataset_dir) or ".", exist_ok=True)

    # Mode 1: Authenticated download via roboflow SDK (preferred, works for private datasets too)
    if api_key:
        try:
            download_with_roboflow_sdk(api_key, workspace, project, version, dataset_dir)
            print("Done. Dataset ready at:", dataset_dir)
            return
        except Exception as exc:
            print(f"Authenticated download failed: {exc}")
            print("Falling back to public direct download...")

    # Mode 2: Public direct download via requests (no API key, may be blocked by Cloudflare)
    zip_path = os.path.join(
        cfg.get("dataset_root", "./data"), f"{project}-v{version}.zip"
    )
    url = get_dataset_download_url(workspace, project, version)

    print(f"Downloading dataset from {url} ...")
    try:
        download_file(url, zip_path)
    except requests.HTTPError as e:
        print(f"Failed to download dataset: {e}")
        print(
            "The dataset may require an API key or the public endpoint has changed.\n"
            "Please either:\n"
            f"  1. Set roboflow_api_key in config.yaml and rerun.\n"
            f"  2. Download manually from: https://universe.roboflow.com/{workspace}/{project}/{version}\n"
            f"     and extract it to: {dataset_dir}"
        )
        raise

    print(f"Extracting to {dataset_dir} ...")
    extract_zip(zip_path, dataset_dir)
    os.remove(zip_path)
    print("Done. Dataset ready at:", dataset_dir)


if __name__ == "__main__":
    main()
