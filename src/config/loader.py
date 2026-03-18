from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _read_yaml_dict(config_path)
    merged = dict(config)
    merged = _apply_profile(
        config_path=config_path,
        config=merged,
        profile_key="evaluation_profile",
        section_key="evaluation",
        profiles_dir_name="evaluations",
    )
    merged = _apply_profile(
        config_path=config_path,
        config=merged,
        profile_key="execution_profile",
        section_key="execution",
        profiles_dir_name="execution",
    )
    return merged


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping in file: {path}")
    return data


def _extract_section(profile_cfg: dict[str, Any], *, section_key: str) -> dict[str, Any]:
    if section_key not in profile_cfg:
        return dict(profile_cfg)
    section_cfg = profile_cfg.get(section_key)
    if section_cfg is None:
        return {}
    if not isinstance(section_cfg, dict):
        raise ValueError(
            f"Profile has invalid '{section_key}' section; expected a mapping."
        )
    return dict(section_cfg)


def _resolve_profile_path(*, config_path: Path, profile_ref: str, profiles_dir_name: str) -> Path:
    ref_path = Path(profile_ref)
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.append(config_path.parent / ref_path)

        profile_name = profile_ref if profile_ref.endswith(".yaml") else f"{profile_ref}.yaml"
        if config_path.parent.name == "experiments" and config_path.parent.parent.name == "configs":
            candidates.append(config_path.parent.parent / profiles_dir_name / profile_name)
        candidates.append(Path("configs") / profiles_dir_name / profile_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve profile "
        f"'{profile_ref}'. Checked: {', '.join(str(p) for p in candidates)}"
    )


def _apply_profile(
    *,
    config_path: Path,
    config: dict[str, Any],
    profile_key: str,
    section_key: str,
    profiles_dir_name: str,
) -> dict[str, Any]:
    profile_ref = config.get(profile_key)
    if not isinstance(profile_ref, str) or not profile_ref.strip():
        return config

    profile_path = _resolve_profile_path(
        config_path=config_path,
        profile_ref=profile_ref.strip(),
        profiles_dir_name=profiles_dir_name,
    )
    profile_cfg = _read_yaml_dict(profile_path)
    profile_section = _extract_section(profile_cfg, section_key=section_key)

    section_override = config.get(section_key, {})
    if section_override is None:
        section_override = {}
    if not isinstance(section_override, dict):
        raise ValueError(f"'{section_key}' must be a mapping when present.")

    merged = dict(config)
    merged[section_key] = _deep_merge_dicts(profile_section, section_override)
    merged[f"_resolved_{profile_key}"] = str(profile_path)
    return merged


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(base_value, value)
        else:
            result[key] = value
    return result
