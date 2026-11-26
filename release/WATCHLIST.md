# Hashed watchlist of hallucinated package names

44491 names that language models recommended in our runs and that were not registered on PyPI or npm at the data freeze (2026-11-23).

Each line is `sha256("slopsquat-watchlist-v1:" + eco + ":" + name)` in hex, where `eco` is `pypi` or `npm` and `name` is normalised (PyPI: lower case, runs of `-_.` replaced by `-`; npm: lower case).

```python
import hashlib
def listed(eco, name, path='watchlist_sha256.txt', salt='slopsquat-watchlist-v1:'):
    import re
    n = re.sub(r'[-_.]+', '-', name).lower() if eco == 'pypi' else name.lower()
    d = hashlib.sha256(f'{salt}{eco}:{n}'.encode()).hexdigest()
    return d in set(open(path).read().split())
```

A match means the name was invented by at least one model in our study; registering or depending on it deserves scrutiny. Absence means nothing: every model invents names we did not observe.
