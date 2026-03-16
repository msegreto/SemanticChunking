from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

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
        total_size_header = response.headers.get("Content-Length")
        total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {destination.name}",
        ) as progress_bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                progress_bar.update(len(chunk))

    return destination


def extract_zip(zip_path: Path, target_dir: Path, remove_archive: bool = True) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    if remove_archive:
        zip_path.unlink(missing_ok=True)


def extract_gzip(gz_path: Path, target_path: Path, remove_archive: bool = False) -> None:
    ensure_dir(target_path.parent)
    with gzip.open(gz_path, "rb") as src, target_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    if remove_archive:
        gz_path.unlink(missing_ok=True)


def write_manifest(target_dir: Path, payload: dict) -> Path:
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def missing_files(base_dir: Path, filenames: Iterable[str]) -> list[str]:
    return [name for name in filenames if not (base_dir / name).exists()]


def _exists_plain_or_gz(base_dir: Path, filename: str) -> bool:
    plain = base_dir / filename
    gz = base_dir / f"{filename}.gz"
    return plain.exists() or gz.exists()


def required_msmarco_files(split: str, include_lookup: bool = False) -> list[str]:
    required = [
        "msmarco-docs.tsv",
    ]
    if include_lookup:
        required.append("msmarco-docs-lookup.tsv")

    if split == "train":
        required += ["msmarco-doctrain-queries.tsv", "msmarco-doctrain-qrels.tsv"]
    elif split == "dev":
        required += ["msmarco-docdev-queries.tsv", "msmarco-docdev-qrels.tsv"]
    elif split == "test":
        required += ["docleaderboard-queries.tsv"]
    else:
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    return required


def list_missing_msmarco_files(base_dir: Path, split: str, include_lookup: bool = False) -> list[str]:
    required = required_msmarco_files(split, include_lookup=include_lookup)
    return [name for name in required if not _exists_plain_or_gz(base_dir, name)]


def validate_msmarco_raw(base_dir: Path, split: str, include_lookup: bool = False) -> bool:
    return len(list_missing_msmarco_files(base_dir, split, include_lookup=include_lookup)) == 0


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


def _required_msmarco_archive_names(split: str, include_lookup: bool = False) -> list[str]:
    names = ["msmarco-docs.tsv.gz"]
    if include_lookup:
        names.append("msmarco-docs-lookup.tsv.gz")

    if split == "train":
        names += ["msmarco-doctrain-queries.tsv.gz", "msmarco-doctrain-qrels.tsv.gz"]
    elif split == "dev":
        names += ["msmarco-docdev-queries.tsv.gz", "msmarco-docdev-qrels.tsv.gz"]
    elif split == "test":
        names += ["docleaderboard-queries.tsv.gz"]
    else:
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    return names


def download_msmarco_documents(
    target_dir: Path,
    split: str,
    force: bool = False,
    include_lookup: bool = False,
    extract_archives: bool = False,
) -> Path:
    target_dir = ensure_dir(target_dir)

    extracted_files: list[str] = []
    archive_names = _required_msmarco_archive_names(split=split, include_lookup=include_lookup)
    selected_urls = {name: MSMARCO_DOC_FILES[name] for name in archive_names}

    for filename, url in selected_urls.items():
        gz_destination = target_dir / filename
        extracted_name = filename[:-3] if filename.endswith(".gz") else filename
        extracted_path = target_dir / extracted_name

        if force or not gz_destination.exists():
            gz_destination.unlink(missing_ok=True)
            download_file(url=url, destination=gz_destination)

        if extract_archives and filename.endswith(".gz"):
            if force or not extracted_path.exists():
                try:
                    extract_gzip(gz_destination, extracted_path, remove_archive=False)
                except (EOFError, OSError):
                    gz_destination.unlink(missing_ok=True)
                    extracted_path.unlink(missing_ok=True)
                    download_file(url=url, destination=gz_destination)
                    extract_gzip(gz_destination, extracted_path, remove_archive=False)
            extracted_files.append(extracted_name)
        else:
            extracted_files.append(filename)

    write_manifest(
        target_dir,
        {
            "dataset_name": "msmarco-docs",
            "source": "official_msmarco",
            "split": split,
            "files_downloaded": list(selected_urls.keys()),
            "files_extracted": extracted_files,
            "include_lookup": include_lookup,
            "extract_archives": extract_archives,
            "urls": selected_urls,
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
