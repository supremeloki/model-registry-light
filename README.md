# model-registry-light

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight ML model registry: versioned artifacts with params, tags, deprecation, and optional JSON persistence — the registry layer for local MLOps without a database server.

## 🚀 Overview

Before models ship they need names, versions, and metadata. `model-registry-light` gives every model an ordered version history (`latest` resolves automatically), validates names against a strict pattern, rejects duplicate versions, and supports tag-based lookup (`find_by_tag("production")`). Deprecation marks versions as non-current instead of deleting history. An optional storage path persists the whole registry as human-readable JSON.

## ✨ Features

- **Strict naming:** `^[a-z0-9][a-z0-9._-]{1,63}$` — no spaces, uppercase, or leading punctuation
- **Versioned history:** append-only registration; `get(name)` returns latest, `get(name, version)` pins
- **Conflict guard:** re-registering the same model+version raises `VersionConflictError`
- **Params & tags:** arbitrary JSON-safe params per version; frozenset tags for fleet queries
- **Deprecation, not deletion:** mark old versions non-current while keeping provenance
- **Clean deletion:** `delete_version` removes one; removing the last deletes the model entry
- **JSON persistence:** pass `storage_path` and the registry survives process restarts
- **Zero dependencies**

## 🚧 Structure

```
model-registry-light/
├── src/model_registry/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/model-registry-light.git
cd model-registry-light
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from pathlib import Path
from model_registry import ModelRegistry

registry = ModelRegistry(storage_path=Path("registry.json"))

registry.register(
    "text-classifier", "1.2.0", "/artifacts/tc-120.pt",
    params={"lr": 0.01, "epochs": 8},
    tags=frozenset({"production"}),
)

latest = registry.get("text-classifier")
print(latest.version, latest.params["lr"])
```

### Tag-based queries

```python
prod_models = registry.find_by_tag("production")
for version in prod_models:
    print(version.model, version.version)
```

## 🔧 Error Handling

```text
RegistryError
├── InvalidModelNameError    # name violates pattern
├── ModelNotFoundError       # unknown model or version
└── VersionConflictError     # duplicate registration
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen dataclasses throughout
- Zero comments — names carry the meaning
- `ruff` clean

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
