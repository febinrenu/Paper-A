"""U6: who claimed the hallucinated names? Classify every claimed package (and claimed controls).

Classes (first match wins):
  malicious      OSV MAL record, or npm "security holding" placeholder (npm's replacement for removed malware)
  placeholder    defensive / reserved / name-protection text
  spam           npm token-farming (tea.xyz) and similar template spam
  empty_squat    effectively no content: one release, tiny artifact, no description, no repository
  organic        repository link and real content (>= 2 releases or a non-trivial artifact)
  unclear        none of the above
Static inspection (never installed, never executed): the latest release artifact is downloaded into
data/quarantine/ and read IN MEMORY as an archive; text members are scanned for indicators
(install-time hooks, obfuscated exec/eval, exfiltration endpoints). Packages with strong indicators
but no OSV record are flagged "suspected_unreported" for disclosure.
Outputs: data/derived/claimants.parquet (name-level, private), tables/u6_claimants.csv.
"""
from __future__ import annotations

import importlib
import io
import json
import re
import tarfile
import zipfile

import httpx
import pandas as pd

from common import DERIVED, QUARANTINE, TABLES, USER_AGENT

st = importlib.import_module("07_stats")

PLACEHOLDER = re.compile(r"(placeholder|reserved|name ?squat|typo ?squat|slop ?squat|hallucinat|defensive|"
                         r"prevent (malicious|abuse)|security (holding|research|purposes)|claimed (by|to protect)|"
                         r"this package (name )?(is|has been) (reserved|claimed)|do not (use|install) this package)", re.I)
SPAM = re.compile(r"(tea\.xyz|tea protocol|tea\.yaml|\bteaxyz\b|airdrop|casino|crypto ?(bonus|reward)|seo backlink|"
                  r"buy followers|free (robux|v-?bucks))", re.I)
INDICATORS = {
    "install_hook_net": re.compile(r"(preinstall|postinstall|install)\"?\s*:\s*\"[^\"]*(curl|wget|node -e|powershell|https?://)", re.I),
    "setup_py_exec": re.compile(r"setup\.py[\s\S]{0,0}|(class\s+\w+\(install\)|cmdclass\s*=)", re.I),
    "obfuscated_exec": re.compile(r"(exec|eval)\s*\(\s*(base64|b64decode|codecs\.decode|bytes\.fromhex|zlib\.decompress|atob|Buffer\.from)", re.I),
    "exfil_endpoint": re.compile(r"(discord(app)?\.com/api/webhooks|api\.telegram\.org/bot|pastebin\.com|ngrok\.io|"
                                 r"interactsh|burpcollaborator|oast\.(fun|live|site|me|pro)|requestbin|webhook\.site|"
                                 r"https?://(?!127\.|0\.0\.0\.0|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)\d{1,3}(\.\d{1,3}){3})", re.I),
    "env_harvest": re.compile(r"(os\.environ|process\.env)\b[\s\S]{0,200}(requests\.post|urllib|fetch\(|https?\.request|axios\.post)", re.I),
    "credential_paths": re.compile(r"(\.ssh/id_rsa|\.aws/credentials|Login Data|Local State|\.npmrc|wallet\.dat|Cookies)", re.I),
}
# code and install manifests only: documentation (README etc.) and vendored dependencies describe
# credential locations legitimately and caused false positives in manual review
TEXT_EXT = (".py", ".js", ".cjs", ".mjs", ".ts", ".json", ".cfg", ".toml", ".sh", ".ps1", ".bat", ".cmd")
SKIP_PATH = re.compile(r"(^|/)(node_modules|vendor|third_party|site-packages|docs?|examples?|tests?)/", re.I)


def classify(r: dict) -> str:
    text = f"{r.get('summary') or ''} {r.get('description_head') or ''}"
    if r.get("mal_post") or r.get("osv_mal") is True or r.get("security_holding") is True:
        return "malicious"
    if PLACEHOLDER.search(text):
        return "placeholder"
    if SPAM.search(text):
        return "spam"
    size = r.get("total_size") if r.get("eco") == "pypi" else r.get("unpacked_size")
    nrel = r.get("n_releases") or 0
    has_repo = r.get("has_repo") is True
    dlen = r.get("description_len") or 0
    if r.get("exists") is True and nrel <= 1 and (size is None or (size or 0) < 4000) and dlen < 150 and not has_repo:
        return "empty_squat"
    if has_repo and (nrel >= 2 or (size or 0) >= 20_000):
        return "organic"
    if r.get("status") == "claimed_removed" or r.get("tombstone") is True:
        return "removed_unknown"
    return "unclear"


def artifact_url(r: dict) -> str | None:
    if r.get("eco") == "npm":
        return r.get("tarball") if isinstance(r.get("tarball"), str) else None
    files = r.get("latest_files")
    try:
        files = json.loads(files) if isinstance(files, str) else files
    except ValueError:
        return None
    if not files:
        return None
    sd = [f for f in files if f.get("type") == "sdist"] or files      # sdist carries setup.py
    return sd[0]["url"] if sd and sd[0].get("size", 0) < 20_000_000 else None


def scan_archive(blob: bytes) -> dict:
    """Read archive members in memory; never extract to disk, never execute."""
    hits = {k: 0 for k in INDICATORS}
    n_text = 0
    members = []
    try:
        if blob[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(blob))
            members = [(i.filename, lambda i=i: z.read(i)) for i in z.infolist() if i.file_size < 2_000_000]
        else:
            t = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
            members = [(m.name, lambda m=m: t.extractfile(m).read()) for m in t.getmembers()
                       if m.isfile() and m.size < 2_000_000]
    except Exception:  # noqa: BLE001
        return {"scan_error": True, **hits}
    for name, read in members[:400]:
        if not name.lower().endswith(TEXT_EXT) or SKIP_PATH.search(name):
            continue
        try:
            text = read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
        n_text += 1
        for k, p in INDICATORS.items():
            if k == "setup_py_exec":
                if name.endswith("setup.py") and p.search(text) and re.search(r"(subprocess|os\.system|urllib|requests|socket)", text):
                    hits[k] += 1
            elif p.search(text):
                hits[k] += 1
    return {"scan_error": False, "n_text_files": n_text, "n_members": len(members), **hits}


def main() -> None:
    cand, _ = st.load_candidates(st.CUTOFF)
    mech = importlib.import_module("10_mechanism")
    h = mech.hallucinated(st.CUTOFF)
    claimed = h[h.claimed | h.mal_post].copy()
    claimed["class"] = [classify(r) for r in claimed.to_dict("records")]

    # static inspection of live artifacts
    cache_p = DERIVED / "claimant_scan_cache.jsonl"
    cache = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[(d["eco"], d["name"])] = d
    qdir = QUARANTINE / "artifacts"
    qdir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True) as c:
        for r in claimed.to_dict("records"):
            key = (r["eco"], r["name"])
            if key in cache or r.get("exists") is not True:
                continue
            url = artifact_url(r)
            res = {"eco": r["eco"], "name": r["name"], "artifact": bool(url)}
            if url:
                try:
                    resp = c.get(url)
                    if resp.status_code == 200:
                        blob = resp.content
                        (qdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{r['eco']}__{r['name']}") + ".bin")).write_bytes(blob)
                        res.update(scan_archive(blob))
                except httpx.HTTPError as ex:
                    res["fetch_error"] = type(ex).__name__
            with open(cache_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            cache[key] = res
    scan = pd.DataFrame(cache.values()) if cache else pd.DataFrame(columns=["eco", "name"])
    claimed = claimed.merge(scan, on=["eco", "name"], how="left")
    strong = ["obfuscated_exec", "exfil_endpoint", "env_harvest", "credential_paths", "install_hook_net", "setup_py_exec"]
    for k in strong:
        if k not in claimed:
            claimed[k] = 0
    claimed["indicator_score"] = claimed[strong].fillna(0).gt(0).sum(axis=1)
    claimed["suspected_unreported"] = (claimed["class"] != "malicious") & (claimed.indicator_score >= 2)
    keep = ["eco", "name", "status", "class", "mal_post", "created", "n_releases", "has_repo", "indicator_score",
            "suspected_unreported", "mentions", "n_models_2024"] + strong
    claimed[[k for k in keep if k in claimed]].to_parquet(DERIVED / "claimants.parquet", index=False)
    out = claimed.groupby(["eco", "class"]).agg(n=("name", "size"), suspected_unreported=("suspected_unreported", "sum"),
                                                with_indicators=("indicator_score", lambda s: int((s > 0).sum()))).reset_index()
    out.to_csv(TABLES / "u6_claimants.csv", index=False)
    print(out.to_string(index=False))
    print("suspected unreported (for disclosure, name-level stays private):", int(claimed.suspected_unreported.sum()))


if __name__ == "__main__":
    main()
