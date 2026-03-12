from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

MSMARCO_DOC_FILES: dict[str, str] = {
    "msmarco-docs.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
    "msmarco-docs-lookup.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz",
    "msmarco-doctrain-queries.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
    "msmarco-doctrain-qrels.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
    "msmarco-docdev-queries.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
    "msmarco-docdev-qrels.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
    "docleaderboard-queries.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz",
}

PUBLIC_BEIR_DATASETS: set[str] = {
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
}

PRIVATE_BEIR_DATASETS: set[str] = {
    "bioasq",
    "robust04",
    "signal1m",
    "trec-news",
}

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


class DatasetDownloadError(RuntimeError):
    pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    ensure_dir(destination.parent)

    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

    return destination


def extract_zip(zip_path: Path, target_dir: Path, remove_archive: bool = True) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    if remove_archive:
        zip_path.unlink(missing_ok=True)


def write_manifest(target_dir: Path, payload: dict) -> Path:
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def missing_files(base_dir: Path, filenames: Iterable[str]) -> list[str]:
    return [name for name in filenames if not (base_dir / name).exists()]


def required_msmarco_files(split: str) -> list[str]:
    required = [
        "msmarco-docs.tsv",
        "msmarco-docs-lookup.tsv",
    ]

    if split == "train":
        required += ["msmarco-doctrain-queries.tsv", "msmarco-doctrain-qrels.tsv"]
    elif split == "dev":
        required += ["msmarco-docdev-queries.tsv", "msmarco-docdev-qrels.tsv"]
    elif split == "test":
        required += ["docleaderboard-queries.tsv"]
    else:
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    return required


def list_missing_msmarco_files(base_dir: Path, split: str) -> list[str]:
    return missing_files(base_dir, required_msmarco_files(split))


def validate_msmarco_raw(base_dir: Path, split: str) -> bool:
    return len(list_missing_msmarco_files(base_dir, split)) == 0


def required_beir_files(split: str) -> list[str]:
    return [
        "corpus.jsonl",
        "queries.jsonl",
        f"qrels/{split}.tsv",
    ]


def list_missing_beir_files(base_dir: Path, split: str) -> list[str]:
    return missing_files(base_dir, required_beir_files(split))


def validate_beir_raw(base_dir: Path, split: str) -> bool:
    return len(list_missing_beir_files(base_dir, split)) == 0


def download_msmarco_documents(target_dir: Path, force: bool = False) -> Path:
    target_dir = ensure_dir(target_dir)

    for filename, url in MSMARCO_DOC_FILES.items():
        destination = target_dir / filename
        if force or not destination.exists():
            download_file(url=url, destination=destination)

    write_manifest(
        target_dir,
        {
            "dataset_name": "msmarco-docs",
            "source": "official_msmarco",
            "files": list(MSMARCO_DOC_FILES.keys()),
            "urls": MSMARCO_DOC_FILES,
        },
    )
    return target_dir


def build_beir_url(dataset_name: str) -> str:
    if dataset_name in PRIVATE_BEIR_DATASETS:
        raise DatasetDownloadError(
            f"BEIR dataset '{dataset_name}' is not publicly downloadable from the standard BEIR mirror."
        )
    if dataset_name not in PUBLIC_BEIR_DATASETS:
        raise DatasetDownloadError(
            f"Unsupported BEIR dataset '{dataset_name}'. Add it explicitly if you want to allow it."
        )
    return BEIR_BASE_URL.format(dataset=dataset_name)


def download_beir_dataset(dataset_name: str, target_root: Path, force: bool = False) -> Path:
    target_root = ensure_dir(target_root)
    dataset_dir = target_root / dataset_name

    # Minimal completeness check
    if not force and (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "queries.jsonl").exists():
        return dataset_dir

    url = build_beir_url(dataset_name)
    archive_path = target_root / f"{dataset_name}.zip"
    download_file(url=url, destination=archive_path)
    extract_zip(zip_path=archive_path, target_dir=target_root, remove_archive=True)

    if not dataset_dir.exists():
        raise DatasetDownloadError(
            f"Downloaded BEIR archive for '{dataset_name}', but extracted directory was not found: {dataset_dir}"
        )

    write_manifest(
        dataset_dir,
        {
            "dataset_name": f"beir/{dataset_name}",
            "source": "beir_public_mirror",
            "url": url,
            "files": ["corpus.jsonl", "queries.jsonl", "qrels/"],
        },
    )
    return dataset_dir