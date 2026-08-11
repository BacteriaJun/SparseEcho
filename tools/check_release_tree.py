from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".md", ".toml", ".json", ".proto", ".h", ".txt", ".yml", ".yaml"}
FORBIDDEN = {
    "non_public_marker": re.compile(r"\b(?:INTERNAL_ONLY|PRIVATE_ONLY|DO_NOT_RELEASE|NOT_FOR_PUBLICATION)\b"),
    "temporary_release_marker": re.compile(r"\b(?:TODO_RELEASE|FIXME_RELEASE|PLACEHOLDER_RELEASE)\b"),
}


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part in {"dist", ".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
    if failures:
        print("release tree check failed")
        for item in failures:
            print(f"- {item}")
        return 1
    print("release tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
