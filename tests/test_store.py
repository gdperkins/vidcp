import hashlib
import os

from vidcp.config import get_settings
from vidcp.store import add_source, artifact_dir, sha256_file

VID = "abcd1234" + "0" * 56  # 64-char id


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "f.bin"
    data = b"hello vidcp"
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_sha256_file_handles_multiple_chunks(tmp_path):
    f = tmp_path / "big.bin"
    data = os.urandom(1024 * 1024 * 2 + 123)  # > 2 MB, not chunk-aligned
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_artifact_dir_structure_and_created():
    d = artifact_dir(VID)
    store = get_settings().store_path
    assert d == store / VID[:2] / VID
    assert d.is_dir()


def test_add_source_copy(tmp_path):
    src = tmp_path / "movie.mp4"
    src.write_bytes(b"video-bytes")
    dest = add_source(src, VID)
    assert dest.exists()
    assert dest.name == "source.mp4"
    assert dest.read_bytes() == b"video-bytes"
    assert dest.parent == artifact_dir(VID)
    assert not dest.samefile(src)  # a copy, distinct inode


def test_add_source_hardlink(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDCP_LINK_MODE", "hardlink")
    get_settings.cache_clear()
    src = tmp_path / "movie.mkv"
    src.write_bytes(b"linked-bytes")
    dest = add_source(src, VID)
    assert dest.exists()
    assert dest.name == "source.mkv"
    assert dest.samefile(src)  # hardlink shares inode
