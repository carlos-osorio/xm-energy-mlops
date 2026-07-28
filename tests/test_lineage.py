"""Tests de src/lineage.py — extracción de hashes de Git y DVC."""
from src.lineage import get_git_commit, get_dvc_hash


def test_git_commit_from_env(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123def")
    assert get_git_commit() == "abc123def"


def test_git_commit_default(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    assert get_git_commit() == "local-dev-commit"


def test_dvc_hash_from_env(monkeypatch):
    monkeypatch.setenv("DVC_HASH", "deadbeef42")
    assert get_dvc_hash() == "deadbeef42"


def test_dvc_hash_fallback_when_no_lockfile(monkeypatch, tmp_path):
    # Sin DVC_HASH y sin dvc.lock accesible -> 'unknown'
    monkeypatch.delenv("DVC_HASH", raising=False)
    monkeypatch.chdir(tmp_path)  # directorio sin dvc.lock
    assert get_dvc_hash() == "unknown"
