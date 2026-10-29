from io import BytesIO

from PIL import Image

from edge_ai.inference import InferenceService, PredictionError


def image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (32, 32), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_demo_prediction_is_deterministic() -> None:
    result = InferenceService.demo().predict_bytes(image_bytes((30, 180, 30)))
    assert result.is_demo
    assert result.label == "healthy"
    assert 0 <= result.confidence <= 1


def test_invalid_image_is_rejected() -> None:
    try:
        InferenceService.demo().predict_bytes(b"not an image")
    except PredictionError as error:
        assert "readable image" in str(error)
    else:
        raise AssertionError("invalid input should fail")

