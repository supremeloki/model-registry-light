import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from model_registry import (
    InvalidModelNameError,
    ModelNotFoundError,
    ModelRegistry,
    VersionConflictError,
)


@pytest.fixture
def registry():
    return ModelRegistry()


def test_register_and_get_latest(registry):
    registry.register("text-classifier", "0.1.0", "/models/tc-01.pt")
    registry.register("text-classifier", "0.2.0", "/models/tc-02.pt")
    latest = registry.get("text-classifier")
    assert latest.version == "0.2.0"
    assert latest.artifact_path == "/models/tc-02.pt"


def test_get_specific_version(registry):
    registry.register("embed", "1.0.0", "/a")
    registry.register("embed", "1.1.0", "/b")
    assert registry.get("embed", "1.0.0").artifact_path == "/a"


def test_unknown_model_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get("ghost")


def test_unknown_version_raises(registry):
    registry.register("mod", "1.0.0", "/x")
    with pytest.raises(ModelNotFoundError):
        registry.get("mod", "9.9")


def test_duplicate_version_rejected(registry):
    registry.register("dup", "1.0", "/a")
    with pytest.raises(VersionConflictError):
        registry.register("dup", "1.0", "/b")


def test_invalid_name_rejected(registry):
    for bad in ["UPPER", "-leading", "", "sp ace"]:
        with pytest.raises(InvalidModelNameError):
            registry.register(bad, "1.0", "/x")


def test_params_and_tags_roundtrip(registry):
    entry = registry.register(
        "ranker", "2.0", "/r",
        params={"lr": 0.01, "layers": 4},
        tags=frozenset({"production", "gpu"}),
    )
    assert entry.params["lr"] == 0.01
    found = registry.find_by_tag("gpu")
    assert len(found) == 1 and found[0].model == "ranker"


def test_deprecate_marks_not_current(registry):
    registry.register("legacy", "0.9", "/old")
    deprecated = registry.deprecate("legacy", "0.9")
    assert deprecated.deprecated is True
    assert deprecated.is_current is False


def test_delete_version_removes_only_target(registry):
    registry.register("mod", "1", "/a")
    registry.register("mod", "2", "/b")
    assert registry.delete_version("mod", "1") is True
    assert registry.get("mod").version == "2"


def test_delete_last_version_removes_model(registry):
    registry.register("solo", "1", "/only")
    registry.delete_version("solo", "1")
    with pytest.raises(ModelNotFoundError):
        registry.get("solo")


def test_list_models_sorted(registry):
    registry.register("zeta", "1", "/z")
    registry.register("alpha", "1", "/a")
    names = [model.name for model in registry.list_models()]
    assert names == ["alpha", "zeta"]


def test_file_persistence(tmp_path):
    store = tmp_path / "registry.json"
    first = ModelRegistry(storage_path=store)
    first.register("net", "3.0", "/n", tags=frozenset({"prod"}))
    first.close if hasattr(first, "close") else None

    reopened = ModelRegistry(storage_path=store)
    assert reopened.get("net", "3.0").artifact_path == "/n"
