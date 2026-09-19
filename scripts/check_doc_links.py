#!/usr/bin/env python3
"""文档链接检查器（只读文档工具，不参与等价性门禁）。

扫描仓库内 Markdown 文件中的相对引用，检查目标是否存在：

* 行内链接：``[text](path)``（忽略 http(s)/mailto/纯锚点/绝对路径）；
* 反引号路径：`` `path/to/file.md` ``，仅取以
  ``.md`` / ``.py`` / ``.yaml`` / ``.json`` 结尾的候选，允许 ``:行号`` 后缀、
  ``#锚点`` 后缀；含通配符、占位符、空格或以 ``/``/``~``/``$`` 开头的路径跳过。

每个候选先相对仓库根解析，再相对所在文件目录解析；两者都不存在时，再按
仓库内文件后缀匹配兜底（文档常用 ``layers/attention.py``、``transforms.py``
这类简写，实际位置在 ``src/fastprove/`` 下）。三者都不命中才算断链。
默认排除缓存目录（``.git``/``__pycache__``/``.pytest_cache``/``.mypy_cache``），
默认包含 ``docs/archive``；用 ``--skip-archive`` 可只看现行文档。

退出码：发现断链返回 1，否则返回 0。报告按文件分组，给出行号与原始目标。

示例：

    python scripts/check_doc_links.py
    python scripts/check_doc_links.py --skip-archive
    python scripts/check_doc_links.py --show-ok            # 同时列出有效链接
    python scripts/check_doc_links.py README.md docs/INDEX.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}
CHECKED_SUFFIXES = (".md", ".py", ".yaml", ".json")
DEFAULT_EXCLUDES = ("docs/archive/**",)

INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+?)\s*(?:\"[^\"]*\")?\)")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?$")
ANCHOR_RE = re.compile(r"#.*$")


@dataclass(frozen=True)
class Reference:
    """一处待检查的引用。"""

    source: Path
    line_number: int
    raw: str
    target: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Markdown 中的相对链接与反引号路径是否存在（只读）。"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="要扫描的 Markdown 文件或目录；默认为仓库根下全部 .md",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="仓库根目录（默认为本脚本上一级目录）",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="跳过 docs/archive/（只检查现行文档）",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="同时打印解析成功的引用（默认只打印断链）",
    )
    return parser.parse_args(argv)


def repo_root(args: argparse.Namespace) -> Path:
    if args.root is not None:
        return Path(args.root).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def discover_markdown(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".md":
            found.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.md")):
                if not any(part in EXCLUDED_DIRS for part in candidate.parts):
                    found.append(candidate)
    return sorted(set(found))


def strip_target(raw: str) -> str | None:
    """把原始引用规整成可检查的路径；不可检查时返回 None。"""

    target = raw.strip().strip("<>")
    if not target or target.startswith(("#", "/", "~", "$", "-")):
        return None
    if "://" in target or target.startswith("mailto:"):
        return None
    if any(char in target for char in (" ", "\t", "(", ")", "*", "?", "{", "}", "<", ">")):
        return None
    target = ANCHOR_RE.sub("", target)
    target = LINE_SUFFIX_RE.sub("", target)
    target = target.rstrip(".,;:!")
    if not target:
        return None
    if target in CHECKED_SUFFIXES or target.startswith("."):
        return None
    return target


def extract_inline(text: str) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    for match in INLINE_LINK_RE.finditer(text):
        target = strip_target(match.group(1))
        if target is None:
            continue
        line_number = text.count("\n", 0, match.start()) + 1
        results.append((line_number, match.group(1).strip(), target))
    return results


def extract_backticks(text: str) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    for match in BACKTICK_RE.finditer(text):
        raw = match.group(1).strip()
        target = strip_target(raw)
        if target is None or not target.endswith(CHECKED_SUFFIXES):
            continue
        line_number = text.count("\n", 0, match.start()) + 1
        results.append((line_number, raw, target))
    return results


def collect_references(source: Path) -> list[Reference]:
    text = source.read_text(encoding="utf-8")
    seen: set[tuple[int, str]] = set()
    references: list[Reference] = []
    for line_number, raw, target in extract_inline(text) + extract_backticks(text):
        key = (line_number, target)
        if key in seen:
            continue
        seen.add(key)
        references.append(Reference(source, line_number, raw, target))
    return references


def build_file_index(root: Path) -> list[str]:
    """建立仓库内可检查文件的相对路径索引（后缀匹配兜底用）。"""

    index: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        try:
            index.append(path.relative_to(root).as_posix())
        except ValueError:
            continue
    return index


def is_broken(reference: Reference, root: Path, file_index: Sequence[str]) -> bool:
    target = reference.target
    candidates = [root / target, reference.source.parent / target]
    if any(candidate.exists() for candidate in candidates):
        return False
    suffix = "/" + target.lstrip("./")
    return not any(
        indexed == target or indexed.endswith(suffix) for indexed in file_index
    )


def format_location(reference: Reference, root: Path) -> str:
    try:
        path = reference.source.resolve().relative_to(root)
    except ValueError:
        path = reference.source
    return f"{path}:{reference.line_number}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = repo_root(args)
    if args.paths:
        targets = [Path(item) if Path(item).is_absolute() else root / item for item in args.paths]
    else:
        targets = [root]
    markdown = discover_markdown(targets)
    if args.skip_archive:
        markdown = [
            path
            for path in markdown
            if not _matches_any(path, root, DEFAULT_EXCLUDES)
        ]
    if not markdown:
        print("check_doc_links: 未找到可扫描的 Markdown 文件", file=sys.stderr)
        return 2

    total = 0
    broken: list[Reference] = []
    ok_count = 0
    file_index = build_file_index(root)
    for source in markdown:
        if any(part in EXCLUDED_DIRS for part in source.parts):
            continue
        for reference in collect_references(source):
            total += 1
            if is_broken(reference, root, file_index):
                broken.append(reference)
            else:
                ok_count += 1
                if args.show_ok:
                    print(f"OK   {format_location(reference, root)} -> {reference.target}")

    for reference in broken:
        print(f"BROKEN {format_location(reference, root)} -> {reference.raw}")

    print(
        f"check_doc_links: 扫描 {len(markdown)} 个 Markdown 文件，"
        f"{total} 条可检查引用，{ok_count} 条有效，{len(broken)} 条断链"
    )
    return 1 if broken else 0


def _matches_any(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    import fnmatch

    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


if __name__ == "__main__":
    raise SystemExit(main())
