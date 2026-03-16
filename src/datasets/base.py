from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDatasetProcessor(ABC):
    """
    Base class for dataset processors.

    Flow:
    1. If normalized data exists and can be used -> load from normalized.
    2. Otherwise ensure raw data is available.
    3. Load from raw.
    4. Save normalized cache (if enabled).
    5. Return normalized in-memory representation.

    Normalized on-disk schema:
        data/normalized/<dataset_name>/
            documents.jsonl
            queries.json
            qrels/<split>.json
            metadata.json
    """

    DEFAULT_SPLIT = "default"
    NORMALIZED_SCHEMA_VERSION = "1.0"
    DEFAULT_SUPPORTED_EXTRINSIC_TASKS = {
        "document_retrieval": True,
        "evidence_retrieval": False,
        "answer_generation": False,
    }
    OPTIONAL_TASK_ARTIFACTS = {
        "evidence_retrieval": ("evidences", "evidences.json"),
        "answer_generation": ("answers", "answers.json"),
    }

    def process(self, config: dict) -> Any:
        normalized_data = self.try_load_reusable_normalized(config)
        if normalized_data is not None:
            return normalized_data

        data = self.build_from_raw(config)
        return self.finalize_raw_data(data, config)

    def try_load_reusable_normalized(self, config: dict) -> dict[str, Any] | None:
        split = self.get_split(config)
        normalized_dir = self.resolve_normalized_dir(config)
        normalized_data = self._try_load_normalized(config=config, normalized_dir=normalized_dir, split=split)

        if normalized_data is not None:
            return normalized_data

        return None

    def build_from_raw(self, config: dict) -> dict[str, Any]:
        raw_path = self.ensure_available(config)
        data = self.load_raw(raw_path=raw_path, config=config)
        optional_payload = self.load_optional_task_artifacts(raw_path=raw_path, config=config)
        if optional_payload:
            data.update(optional_payload)
        self.validate_dataset_payload(data, context="raw load")
        return data

    def finalize_raw_data(self, data: dict[str, Any], config: dict) -> dict[str, Any]:
        split = self.get_split(config)
        normalized_dir = self.resolve_normalized_dir(config)
        raw_path = self.resolve_raw_dir(config)
        required_tasks = self.required_extrinsic_tasks(config)
        checked_tasks = sorted({*required_tasks, "document_retrieval"})
        supported_tasks = self.supported_extrinsic_tasks(config)
        if "evidences" in data:
            supported_tasks["evidence_retrieval"] = True
        if "answers" in data:
            supported_tasks["answer_generation"] = True

        data.setdefault("metadata", {})
        data["metadata"].update(
            {
                "loaded_from": "raw",
                "raw_path": str(raw_path),
                "normalized_path": str(normalized_dir),
                "split": split,
                "normalized_schema_version": self.NORMALIZED_SCHEMA_VERSION,
                "supported_extrinsic_tasks": supported_tasks,
                "checked_extrinsic_tasks": checked_tasks,
                "raw_refresh_attempted_for_tasks": checked_tasks,
            }
        )

        if self.should_save_normalized(config):
            self.validate_dataset_payload(data, context="pre-save normalized")
            self.save_normalized(data=data, normalized_dir=normalized_dir, split=split)
            normalized_files = {
                "documents": "documents.jsonl",
                "queries": "queries.json",
                "qrels": f"qrels/{split}.json",
            }
            if "evidences" in data:
                normalized_files["evidences"] = "evidences.json"
            if "answers" in data:
                normalized_files["answers"] = "answers.json"
            data["metadata"].update(
                {
                    "normalized_files": normalized_files
                }
            )

            if self.should_delete_raw_after_normalized(config):
                self.cleanup_raw(raw_path)

        return data

    def _try_load_normalized(self, config: dict, normalized_dir: Path, split: str) -> dict[str, Any] | None:
        if not self.should_use_normalized(config):
            return None

        if not self.normalized_exists(normalized_dir, split):
            return None

        if not self.is_normalized_compatible(normalized_dir=normalized_dir, split=split, config=config):
            return None

        data = self.load_normalized(normalized_dir=normalized_dir, split=split, config=config)
        self._try_enrich_normalized_task_artifacts(
            data=data,
            normalized_dir=normalized_dir,
            config=config,
        )

        required_tasks = self.required_extrinsic_tasks(config)
        supported_tasks_after_enrich = self.supported_extrinsic_tasks(config)
        if "evidences" in data:
            supported_tasks_after_enrich["evidence_retrieval"] = True
        if "answers" in data:
            supported_tasks_after_enrich["answer_generation"] = True
        missing_required_tasks = [
            task_name
            for task_name in required_tasks
            if not bool(supported_tasks_after_enrich.get(task_name, False))
        ]
        attempted_raw_refresh = self._normalized_raw_refresh_attempted_tasks(normalized_dir)
        should_force_raw_refresh = [
            task_name
            for task_name in missing_required_tasks
            if task_name not in attempted_raw_refresh
        ]
        if should_force_raw_refresh:
            print(
                "[DATASET] Missing required task artifacts in normalized; "
                "forcing one raw refresh attempt for tasks="
                f"{should_force_raw_refresh}."
            )
            return None

        self.validate_dataset_payload(data, context="normalized load")
        checked_tasks = sorted({*required_tasks, "document_retrieval"})
        supported_tasks = supported_tasks_after_enrich
        data.setdefault("metadata", {})
        data["metadata"].update(
            {
                "loaded_from": "normalized",
                "normalized_path": str(normalized_dir),
                "split": split,
                "supported_extrinsic_tasks": supported_tasks,
                "checked_extrinsic_tasks": checked_tasks,
            }
        )
        self._persist_normalized_task_state(
            data=data,
            normalized_dir=normalized_dir,
            split=split,
        )
        return data

    def supported_extrinsic_tasks(self, config: dict) -> dict[str, bool]:
        supported = dict(self.DEFAULT_SUPPORTED_EXTRINSIC_TASKS)
        return {str(k): bool(v) for k, v in supported.items()}

    def required_extrinsic_tasks(self, config: dict) -> list[str]:
        raw = config.get("required_extrinsic_tasks", ["document_retrieval"])
        if not isinstance(raw, list):
            return ["document_retrieval"]
        tasks: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            name = item.strip().lower().replace("-", "_")
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            tasks.append(name)
        if "document_retrieval" not in seen:
            tasks.insert(0, "document_retrieval")
        return tasks

    def validate_dataset_payload(self, data: Any, context: str = "dataset payload") -> None:
        if not isinstance(data, dict):
            raise TypeError(f"Invalid {context}: expected dict, got {type(data).__name__}.")

        required_keys = {"documents", "queries", "qrels", "metadata"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Invalid {context}: missing required keys {sorted(missing_keys)}."
            )

        documents = data["documents"]
        queries = data["queries"]
        qrels = data["qrels"]
        metadata = data["metadata"]

        if not isinstance(documents, dict):
            raise TypeError(
                f"Invalid {context}: 'documents' must be a dict, got {type(documents).__name__}."
            )
        if not isinstance(queries, dict):
            raise TypeError(
                f"Invalid {context}: 'queries' must be a dict, got {type(queries).__name__}."
            )
        if not isinstance(qrels, dict):
            raise TypeError(
                f"Invalid {context}: 'qrels' must be a dict, got {type(qrels).__name__}."
            )
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Invalid {context}: 'metadata' must be a dict, got {type(metadata).__name__}."
            )

        self._validate_documents(documents, context=context)
        self._validate_queries(queries, context=context)
        self._validate_qrels(qrels, context=context)

    def get_split(self, config: dict) -> str:
        return str(config.get("split", self.DEFAULT_SPLIT))

    def resolve_raw_dir(self, config: dict) -> Path:
        if config.get("data_dir"):
            return Path(config["data_dir"]).expanduser().resolve()

        dataset_name = str(config.get("name", "default")).strip("/")
        return Path("data/raw") / dataset_name

    def resolve_normalized_dir(self, config: dict) -> Path:
        if config.get("normalized_dir"):
            return Path(config["normalized_dir"]).expanduser().resolve()

        dataset_name = str(config.get("name", "default")).strip("/")
        return Path("data/normalized") / dataset_name

    def should_download(self, config: dict) -> bool:
        return bool(config.get("download_if_missing", True))

    def should_use_normalized(self, config: dict) -> bool:
        # If max_documents is set, avoid accidentally reusing a full-cache normalized
        # for a partial/debug run.
        if config.get("max_documents") is not None:
            return False
        if bool(config.get("force_rebuild_normalized", False)):
            return False
        return bool(config.get("use_normalized_if_available", True))

    def should_save_normalized(self, config: dict) -> bool:
        if config.get("max_documents") is not None:
            return False
        return bool(config.get("save_normalized", True))

    def should_delete_raw_after_normalized(self, config: dict) -> bool:
        return bool(config.get("delete_raw_after_normalized", True))

    def normalized_exists(self, normalized_dir: Path, split: str) -> bool:
        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        metadata_path = normalized_dir / "metadata.json"

        return (
            documents_path.exists()
            and queries_path.exists()
            and qrels_path.exists()
            and metadata_path.exists()
        )

    def is_normalized_compatible(self, normalized_dir: Path, split: str, config: dict) -> bool:
        metadata_path = normalized_dir / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(metadata, dict):
            return False

        dataset_name = metadata.get("dataset_name")
        expected_names = self.expected_dataset_names(config)
        if dataset_name not in expected_names:
            return False

        if metadata.get("split") != split:
            return False

        if metadata.get("normalized_schema_version") != self.NORMALIZED_SCHEMA_VERSION:
            return False

        normalized_files = metadata.get("normalized_files")
        if not isinstance(normalized_files, dict):
            return False

        if normalized_files.get("documents") != "documents.jsonl":
            return False
        if normalized_files.get("queries") != "queries.json":
            return False
        if normalized_files.get("qrels") != f"qrels/{split}.json":
            return False

        return True

    def expected_dataset_names(self, config: dict) -> set[str]:
        return {str(config.get("name", "default")).strip()}

    def load_normalized(self, normalized_dir: Path, split: str, config: dict) -> Any:
        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        metadata_path = normalized_dir / "metadata.json"

        if not self.normalized_exists(normalized_dir, split):
            raise FileNotFoundError(
                f"Normalized dataset not complete for split='{split}' at '{normalized_dir}'"
            )

        documents = self._read_documents_jsonl(documents_path)
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        normalized_files = metadata.get("normalized_files", {})

        evidences = None
        answers = None
        if isinstance(normalized_files, dict):
            evidences_rel = normalized_files.get("evidences")
            answers_rel = normalized_files.get("answers")
            if isinstance(evidences_rel, str) and evidences_rel:
                evidence_path = normalized_dir / evidences_rel
                if evidence_path.exists():
                    evidences = json.loads(evidence_path.read_text(encoding="utf-8"))
            if isinstance(answers_rel, str) and answers_rel:
                answers_path = normalized_dir / answers_rel
                if answers_path.exists():
                    answers = json.loads(answers_path.read_text(encoding="utf-8"))

        metadata.update(
            {
                "dataset_name": config.get("name", metadata.get("dataset_name")),
                "split": split,
            }
        )

        payload = {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "metadata": metadata,
        }
        if evidences is not None:
            payload["evidences"] = evidences
        if answers is not None:
            payload["answers"] = answers
        return payload

    def save_normalized(self, data: dict, normalized_dir: Path, split: str) -> None:
        documents = data.get("documents", {})
        queries = data.get("queries", {})
        qrels = data.get("qrels", {})
        evidences = data.get("evidences")
        answers = data.get("answers")
        metadata = dict(data.get("metadata", {}))

        if documents is None:
            raise ValueError(
                "Cannot save normalized dataset because 'documents' is None. "
                "For now, processors should fully load documents before normalization."
            )

        self._ensure_dir(normalized_dir)
        self._ensure_dir(normalized_dir / "qrels")

        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        evidences_path = normalized_dir / "evidences.json"
        answers_path = normalized_dir / "answers.json"
        metadata_path = normalized_dir / "metadata.json"

        self._write_documents_jsonl(documents_path, documents)
        queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
        qrels_path.write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")
        normalized_files = {
            "documents": "documents.jsonl",
            "queries": "queries.json",
            "qrels": f"qrels/{split}.json",
        }
        if evidences is not None:
            evidences_path.write_text(json.dumps(evidences, ensure_ascii=False, indent=2), encoding="utf-8")
            normalized_files["evidences"] = "evidences.json"
        if answers is not None:
            answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
            normalized_files["answers"] = "answers.json"

        metadata.update(
            {
                "normalized_schema_version": self.NORMALIZED_SCHEMA_VERSION,
                "normalized_files": normalized_files,
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_optional_task_artifacts(self, raw_path: Path, config: dict) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        required_tasks = self.required_extrinsic_tasks(config)
        for task_name in required_tasks:
            mapping = self.OPTIONAL_TASK_ARTIFACTS.get(task_name)
            if mapping is None:
                continue
            payload_key, filename = mapping
            artifact_path = raw_path / filename
            if not artifact_path.exists():
                continue
            try:
                artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                payload[payload_key] = artifact_payload
            except Exception as e:
                print(
                    "[DATASET] Optional artifact found but unreadable for "
                    f"{task_name} at {artifact_path}: {e}. Ignoring artifact."
                )
        return payload

    def _try_enrich_normalized_task_artifacts(
        self,
        *,
        data: dict[str, Any],
        normalized_dir: Path,
        config: dict,
    ) -> None:
        required_tasks = self.required_extrinsic_tasks(config)

        for task_name in required_tasks:
            mapping = self.OPTIONAL_TASK_ARTIFACTS.get(task_name)
            if mapping is None:
                continue
            payload_key, filename = mapping
            if payload_key in data:
                continue

            destination = normalized_dir / filename

            if destination.exists():
                try:
                    data[payload_key] = json.loads(destination.read_text(encoding="utf-8"))
                    continue
                except Exception as e:
                    print(
                        "[DATASET] Optional normalized artifact unreadable for "
                        f"{task_name} at {destination}: {e}. Continuing."
                    )

    def _persist_normalized_task_state(
        self,
        *,
        data: dict[str, Any],
        normalized_dir: Path,
        split: str,
    ) -> None:
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        normalized_files = metadata.get("normalized_files", {})
        if not isinstance(normalized_files, dict):
            normalized_files = {}
        normalized_files = dict(normalized_files)
        normalized_files.setdefault("documents", "documents.jsonl")
        normalized_files.setdefault("queries", "queries.json")
        normalized_files.setdefault("qrels", f"qrels/{split}.json")

        if "evidences" in data:
            evidences_path = normalized_dir / "evidences.json"
            evidences_path.parent.mkdir(parents=True, exist_ok=True)
            evidences_path.write_text(
                json.dumps(data["evidences"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            normalized_files["evidences"] = "evidences.json"

        if "answers" in data:
            answers_path = normalized_dir / "answers.json"
            answers_path.parent.mkdir(parents=True, exist_ok=True)
            answers_path.write_text(
                json.dumps(data["answers"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            normalized_files["answers"] = "answers.json"

        metadata["normalized_files"] = normalized_files
        metadata_path = normalized_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalized_raw_refresh_attempted_tasks(self, normalized_dir: Path) -> set[str]:
        metadata_path = normalized_dir / "metadata.json"
        if not metadata_path.exists():
            return set()
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        if not isinstance(metadata, dict):
            return set()
        raw_attempted = metadata.get("raw_refresh_attempted_for_tasks", [])
        if not isinstance(raw_attempted, list):
            return set()
        return {
            str(task).strip().lower().replace("-", "_")
            for task in raw_attempted
            if isinstance(task, str) and task.strip()
        }

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_documents_jsonl(path: Path, documents: dict[str, dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file_obj:
            for doc_id, payload in documents.items():
                row = {"id": doc_id, **payload}
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_documents_jsonl(path: Path) -> dict[str, dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.pop("id")
                documents[doc_id] = row
        return documents

    @staticmethod
    def cleanup_raw(raw_path: Path) -> None:
        if not raw_path.exists():
            return

        if raw_path.is_dir():
            shutil.rmtree(raw_path)
            return

        raw_path.unlink()

    @staticmethod
    def _validate_documents(documents: dict[Any, Any], context: str) -> None:
        for doc_id, payload in documents.items():
            if not isinstance(doc_id, str):
                raise TypeError(
                    f"Invalid {context}: document id must be str, got {type(doc_id).__name__}."
                )
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Invalid {context}: document '{doc_id}' payload must be a dict, "
                    f"got {type(payload).__name__}."
                )

    @staticmethod
    def _validate_queries(queries: dict[Any, Any], context: str) -> None:
        for query_id, text in queries.items():
            if not isinstance(query_id, str):
                raise TypeError(
                    f"Invalid {context}: query id must be str, got {type(query_id).__name__}."
                )
            if not isinstance(text, str):
                raise TypeError(
                    f"Invalid {context}: query '{query_id}' text must be str, "
                    f"got {type(text).__name__}."
                )

    @staticmethod
    def _validate_qrels(qrels: dict[Any, Any], context: str) -> None:
        for query_id, doc_map in qrels.items():
            if not isinstance(query_id, str):
                raise TypeError(
                    f"Invalid {context}: qrels query id must be str, got {type(query_id).__name__}."
                )
            if not isinstance(doc_map, dict):
                raise TypeError(
                    f"Invalid {context}: qrels entry for query '{query_id}' must be a dict, "
                    f"got {type(doc_map).__name__}."
                )

            for doc_id, relevance in doc_map.items():
                if not isinstance(doc_id, str):
                    raise TypeError(
                        f"Invalid {context}: qrels doc id must be str, got {type(doc_id).__name__}."
                    )
                if not isinstance(relevance, int):
                    raise TypeError(
                        f"Invalid {context}: qrels relevance for query '{query_id}', "
                        f"doc '{doc_id}' must be int, got {type(relevance).__name__}."
                    )

    @abstractmethod
    def ensure_available(self, config: dict) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, raw_path: Path, config: dict) -> Any:
        raise NotImplementedError
