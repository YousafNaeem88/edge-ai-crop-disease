import json

from PIL import Image

from edge_ai.audit import audit_dataset


def test_audit_reports_classes_and_invalid_files(tmp_path) -> None:
    healthy = tmp_path / "raw" / "healthy"
    diseased = tmp_path / "raw" / "diseased"
    healthy.mkdir(parents=True)
    diseased.mkdir(parents=True)
    Image.new("RGB", (20, 20), (10, 100, 10)).save(healthy / "one.png")
    Image.new("RGB", (30, 20), (100, 10, 10)).save(diseased / "one.png")
    (healthy / "broken.png").write_bytes(b"bad")
    output = tmp_path / "audit.json"
    result = audit_dataset(tmp_path / "raw", output)
    assert result["total_valid_images"] == 2
    assert result["invalid_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["class_count"] == 2
