from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Mapping

from agent.config import PROMPT_VARIANT


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prompts_dir() -> Path:
    return _project_root() / "prompts"


@lru_cache(maxsize=128)
def _load_prompt_text_cached(cache_key: str) -> str:
    variant, name = cache_key.split("|", 1)
    prompts_dir = _prompts_dir()
    path_name = Path(name)
    candidates: list[Path] = []

    def add(root: Path) -> None:
        if path_name.suffix:
            candidates.append(root / name)
        else:
            candidates.extend([root / f"{name}.txt", root / f"{name}.md"])

    if variant:
        add(prompts_dir / variant)
    add(prompts_dir)

    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: {name}. Tried: {', '.join(map(str, candidates))}")


def load_prompt_text(name: str) -> str:
    return _load_prompt_text_cached(f"{PROMPT_VARIANT}|{name}")


def render_prompt(name: str, variables: Mapping[str, object] | None = None) -> str:
    text = load_prompt_text(name)
    if not variables:
        return text
    mapping = {key: "" if value is None else str(value) for key, value in variables.items()}
    return Template(text).safe_substitute(mapping)
