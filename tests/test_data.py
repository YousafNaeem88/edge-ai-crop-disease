from pathlib import Path

from PIL import Image

from edge_ai.data import build_manifest


def test_manifest_is_reproducible_and_validates_images(tmp_path: Path) -> None:
    for label, colour in [("healthy", (30, 180, 30)), ("diseased", (180, 30, 30))]:
        directory = tmp_path / "raw" / label
        directory.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (24, 24), colour).save(directory / f"{index}.png")
    (tmp_path / "raw" / "healthy" / "bad.png").write_bytes(b"invalid")
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    build_manifest(tmp_path / "raw", first, seed=7)
    build_manifest(tmp_path / "raw", second, seed=7)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert len(first.read_text(encoding="utf-8").splitlines()) == 9
