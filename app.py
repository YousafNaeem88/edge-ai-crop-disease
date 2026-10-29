import json
from pathlib import Path

import streamlit as st

from edge_ai.inference import InferenceService, PredictionError
from edge_ai.storage import HistoryStore

ARTIFACT = Path("artifacts/model.pt")
DB_PATH = Path("data/history.sqlite3")

st.set_page_config(page_title="Crop Disease Edge-AI", page_icon="🌿", layout="wide")

st.title("🌿 Crop Disease Edge-AI")
st.caption("Offline-capable image classification prototype")
st.warning(
    "This system provides an AI prediction for research and demonstration. "
    "It is not a confirmed agricultural diagnosis or treatment recommendation."
)

store = HistoryStore(DB_PATH)
load_error = None
if ARTIFACT.exists():
    try:
        service = InferenceService.from_checkpoint(ARTIFACT)
    except PredictionError as exc:
        service = InferenceService.demo()
        load_error = str(exc)
else:
    service = InferenceService.demo()

with st.sidebar:
    st.header("System status")
    if service.is_demo:
        st.info("Demo mode: no trained checkpoint is installed.")
        if load_error:
            st.error(f"Checkpoint fallback: {load_error}")
        st.code("python -m edge_ai.train ...", language="powershell")
    else:
        st.success(f"Model loaded: {service.model_name}")
    page = st.radio("Navigate", ["Predict", "History", "Results", "About"])

if page == "Predict":
    st.subheader("Local prediction")
    uploaded = st.file_uploader("Upload a crop or leaf image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Input image", use_container_width=True)
        if st.button("Run local inference", type="primary"):
            try:
                result = service.predict_bytes(uploaded.getvalue())
                st.metric("Predicted class", result.label)
                st.metric("Confidence", f"{result.confidence:.1%}")
                st.caption(f"Model: {result.model_name} · Latency: {result.latency_ms:.1f} ms")
                if result.is_demo:
                    st.info("This is a deterministic demo heuristic, not a trained model result.")
                store.add(result.label, result.confidence, result.model_name)
            except PredictionError as exc:
                st.error(str(exc))
elif page == "History":
    st.subheader("Local prediction history")
    rows = store.list(limit=100)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if st.button("Delete all local history"):
            store.clear()
            st.rerun()
    else:
        st.info("No local predictions have been recorded yet.")
elif page == "Results":
    st.subheader("Experiment results")
    result_files = sorted(Path("results").glob("*.json"))
    if not result_files:
        st.info("No evaluation JSON found. Run the evaluation command after training a checkpoint.")
        st.code(
            "python -m edge_ai.evaluate --checkpoint artifacts/model.pt "
            "--manifest data/manifest.csv --output results/model.json",
            language="powershell",
        )
    else:
        for result_file in result_files:
            st.markdown(f"**{result_file.name}**")
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                st.error(f"Could not read {result_file.name}: {exc}")
                continue
            metrics = {
                key: result[key]
                for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
                if key in result
            }
            if metrics:
                st.dataframe([metrics], use_container_width=True, hide_index=True)
            st.json(result)
else:
    st.subheader("About this prototype")
    st.markdown(
        """
        The required prediction path is local: image preprocessing, model inference, and
        SQLite history operate on this computer without an external AI API. Use the
        documented training and evaluation commands to replace demo mode with a real
        ResNet18 or MobileNetV3-Small checkpoint.
        """
    )
    st.write("Database:", str(DB_PATH))
    st.write("Checkpoint:", str(ARTIFACT))
