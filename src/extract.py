"""Unified package-name extraction from model outputs (all models share one parser).

Follows Spracklen et al.'s normalisation and filter lists (PEP 503 normalisation for Python,
Node core-module list, npm option list, published false-positive lists) but replaces their
per-model ad-hoc parsers with a single deterministic extractor so that all ten models are
measured identically.
"""
from __future__ import annotations

import csv
import re
import sys
from functools import lru_cache

from common import SPRACKLEN

PY_NAME = re.compile(r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$")
NPM_NAME = re.compile(r"^(@[a-z0-9~][a-z0-9._~-]*/)?[a-z0-9~][a-z0-9._~-]*$")
STOP = {"none", "n/a", "na", "null", "pip", "pip3", "npm", "yarn", "pnpm", "npx", "python", "python3", "javascript",
        "js", "node", "nodejs", "node.js", "typescript", "and", "or", "the", "a", "an", "etc", "optional", "packages",
        "package", "module", "modules", "library", "libraries", "install", "required", "no", "standard", "builtin",
        "built-in", "stdlib", "nothing", "only", "with", "for", "using", "in", "of", "to", "is", "are", "not", "any",
        "code", "bash", "sh", "shell", "cmd", "html", "css", "json", "sql", "java", "php", "c++", "c#", "go", "ruby"}
PIP_OPTS_WITH_ARG = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url",
                     "--extra-index-url", "-f", "--find-links", "-t", "--target", "--prefix", "--root", "--platform",
                     "--python-version", "--implementation", "--abi", "--trusted-host", "--src", "--upgrade-strategy"}


def normalize_py(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@lru_cache(maxsize=None)
def node_core() -> frozenset:
    with open(SPRACKLEN / "Data/Javascript/core_modules.csv", encoding="utf-8") as f:
        mods = [m.strip() for row in csv.reader(f) for m in row]
    extra = {"node:" + m for m in mods}
    return frozenset(mods) | frozenset(extra) | {"test", "sqlite", "sea"}


@lru_cache(maxsize=None)
def py_stdlib() -> frozenset:
    return frozenset(normalize_py(m) for m in sys.stdlib_module_names) | {"tkinter", "distutils", "imp", "asyncore", "asynchat"}


@lru_cache(maxsize=None)
def false_positives(lang: str) -> frozenset:
    out = set()
    with open(SPRACKLEN / f"Data/{lang}/false_positive_packages.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) > 1:
                out.add(normalize_py(row[1]) if lang == "Python" else row[1].lower())
    return frozenset(out)


def _clean_token(tok: str) -> str:
    tok = tok.strip().strip("`'\"*_ ").strip()
    tok = re.sub(r"^(\d+[.)]|[-*•])\s*", "", tok)          # list bullets / numbering
    tok = re.sub(r"\s*\(.*?\)\s*$", "", tok)                     # trailing "(optional)" notes
    tok = tok.strip().strip("`'\"*_.,;: ").strip()
    return tok


def from_list_answer(text: str, lang: str) -> list[tuple[str, str]]:
    """Parse a 'comma separated list of packages' answer (steps t1 / t2). Returns (raw, name)."""
    if not text:
        return []
    text = re.sub(r"```[a-zA-Z]*", "", text)
    parts = re.split(r"[,\n]", text)
    out = []
    for p in parts:
        tok = _clean_token(p)
        if not tok or " " in tok or len(tok) > 100:
            continue
        out.append(tok)
    return [(t, n) for t, n in ((t, canonical(t, lang)) for t in out) if n]


def canonical(tok: str, lang: str) -> str | None:
    tok = tok.strip()
    if lang == "Python":
        tok = re.split(r"[=<>!~;\[@ ]", tok, maxsplit=1)[0]        # drop version spec / extras / markers
        if normalize_py(tok.split(".")[0]) in py_stdlib():        # e.g. http.server, os.path
            return None
        n = normalize_py(tok)
        if n in STOP or not PY_NAME.match(n) or len(n) < 2:
            return None
        if n in py_stdlib():
            return None
        return n
    tok = tok.lower()
    if tok.startswith("@"):
        parts = tok.split("/")
        if len(parts) < 2:
            return None
        tok = "/".join(parts[:2])
    else:
        tok = tok.split("/")[0]
    tok = re.sub(r"@[\^~><=]?[\w.\-*]+$", "", tok) if not tok.startswith("@") else re.sub(r"(?<=.)@[\^~><=]?[\w.\-*]+$", "", tok)
    if tok in STOP or tok in node_core() or tok.startswith((".", "node:", "http:", "https:", "file:", "git+")):
        return None
    if not NPM_NAME.match(tok) or len(tok) > 214 or len(tok) < 2:
        return None
    return tok


PIP_RE = re.compile(r"\bpip3?\s+install\s+([^\n#;&|`]+)", re.I)
JS_INSTALL_RE = re.compile(r"\b(?:npm\s+(?:install|i|add)|yarn\s+(?:global\s+)?add|pnpm\s+(?:add|install|i)|npx)\s+([^\n#;&|`]+)", re.I)
REQUIRE_RE = re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
IMPORT_FROM_RE = re.compile(r"\bimport\s+(?:[\w*{}\s,]+?\s+from\s+)?['\"]([^'\"]+)['\"]")
DYN_IMPORT_RE = re.compile(r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
EXPORT_FROM_RE = re.compile(r"\bexport\s+[\w*{}\s,]+?\s+from\s+['\"]([^'\"]+)['\"]")


def _install_args(argstr: str, with_arg: set) -> list[str]:
    toks, out, skip = argstr.split(), [], False
    for t in toks:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            if t.split("=")[0] in with_arg and "=" not in t:
                skip = True
            continue
        if any(s in t for s in ("/", "\\", "http", ".txt", ".whl", ".tar", "$", "{", "<")) and not t.startswith("@"):
            continue
        out.append(t)
    return out


def from_code(text: str, lang: str) -> list[tuple[str, str]]:
    """Packages referenced by install commands (both langs) and import specifiers (JS only). Returns (raw, name)."""
    if not text:
        return []
    names = []
    if lang == "Python":
        for m in PIP_RE.finditer(text):
            names += _install_args(m.group(1), PIP_OPTS_WITH_ARG)
    else:
        for m in JS_INSTALL_RE.finditer(text):
            names += _install_args(m.group(1), {"--registry", "--prefix", "-w", "--workspace", "--filter"})
        for rx in (REQUIRE_RE, IMPORT_FROM_RE, DYN_IMPORT_RE, EXPORT_FROM_RE):
            names += rx.findall(text)
    return [(t, n) for t, n in ((t, canonical(t, lang)) for t in names) if n]


def is_dotted_path(raw: str, lang: str) -> bool:
    """Python module path written as a package (e.g. 'matplotlib.pyplot', 'django.db.models')."""
    return lang == "Python" and "." in raw.strip("`'\" ").rstrip(".")


def extract_row(row: dict) -> list[tuple[str, str, bool]]:
    """Return [(source, name, dotted)] for one generation row; filters published false positives."""
    lang = row["lang"]
    fp = false_positives(lang)
    found, seen = [], set()
    for src, fn, field in (("code", from_code, "code"), ("t1", from_list_answer, "t1"), ("t2", from_list_answer, "t2")):
        for raw, n in fn(row.get(field) or "", lang):
            if n in fp or (src, n) in seen:
                continue
            seen.add((src, n))
            found.append((src, n, is_dotted_path(raw, lang)))
    return found
