from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None

from src.config import CLASS_NAMES, MODELS_DIR, RUNS_DIR
from src.detector import ProductDetector, draw_overlay
from src.tracker import CentroidTracker, LineCounter


MODEL_CANDIDATES = [
    MODELS_DIR / "best.pt",
    RUNS_DIR / "beverage_yolov8n" / "weights" / "best.pt",
]
DEFAULT_MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
RTC_CONFIGURATION = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if RTCConfiguration
    else None
)


def init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("tracker", CentroidTracker())
    st.session_state.setdefault("counter", LineCounter(CLASS_NAMES))
    st.session_state.setdefault("unknown_logs", [])
    st.session_state.setdefault("chart_rows", [])


def reset_runtime() -> None:
    st.session_state.tracker = CentroidTracker()
    st.session_state.counter = LineCounter(CLASS_NAMES)
    st.session_state.unknown_logs = []
    st.session_state.chart_rows = []


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, class_names: tuple[str, ...]) -> ProductDetector:
    return ProductDetector(model_path, list(class_names))


def sidebar() -> dict:
    st.sidebar.header("Konfigurasi")
    model_path = st.sidebar.text_input("Path model YOLO", value=str(DEFAULT_MODEL_PATH))
    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
    line_position = st.sidebar.slider("Posisi garis counting (%)", 10, 90, 55, 1)
    min_area = st.sidebar.number_input("Minimum area bbox", min_value=100, max_value=100_000, value=1_500, step=100)
    camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    return {
        "model_path": Path(model_path),
        "confidence": confidence,
        "line_position": line_position,
        "min_area": int(min_area),
        "camera_index": int(camera_index),
    }


def render_metrics() -> None:
    counter: LineCounter = st.session_state.counter
    total_known = sum(counter.counts.values())
    total_all = total_known + counter.unknown_count
    error_rate = (counter.unknown_count / total_all) if total_all else 0.0

    cols = st.columns(4)
    cols[0].metric("Total produk", total_known)
    cols[1].metric("Unknown", counter.unknown_count)
    cols[2].metric("Total objek", total_all)
    cols[3].metric("Error rate", f"{error_rate:.2%}")

    count_cols = st.columns(max(1, len(counter.counts)))
    for idx, (name, value) in enumerate(counter.counts.items()):
        count_cols[idx % len(count_cols)].metric(name, value)


def render_logs() -> None:
    logs = st.session_state.unknown_logs[-50:]
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada objek unknown.")


def render_state_metrics(counts: dict[str, int], unknown_count: int) -> None:
    total_known = sum(counts.values())
    total_all = total_known + unknown_count
    error_rate = (unknown_count / total_all) if total_all else 0.0

    cols = st.columns(4)
    cols[0].metric("Total produk", total_known)
    cols[1].metric("Unknown", unknown_count)
    cols[2].metric("Total objek", total_all)
    cols[3].metric("Error rate", f"{error_rate:.2%}")

    count_cols = st.columns(max(1, len(counts)))
    for idx, (name, value) in enumerate(counts.items()):
        count_cols[idx % len(count_cols)].metric(name, value)


class BrowserCameraProcessor(VideoProcessorBase):
    def __init__(self, detector: ProductDetector, settings: dict, class_names: list[str]) -> None:
        self.detector = detector
        self.settings = settings
        self.tracker = CentroidTracker()
        self.counter = LineCounter(class_names)
        self.unknown_logs: list[dict] = []
        self.chart_rows: list[dict] = []
        self.lock = Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        line_y = int(image.shape[0] * self.settings["line_position"] / 100)
        detections, unknowns = self.detector.detect(
            image,
            confidence_threshold=self.settings["confidence"],
            min_area=self.settings["min_area"],
        )
        tracks = self.tracker.update(detections)
        self.counter.apply(tracks, line_y)

        with self.lock:
            for unknown in unknowns:
                self.unknown_logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": round(unknown["confidence"], 3),
                        "reason": unknown["reason"],
                        "area": unknown["area"],
                        "aspect_ratio": round(unknown["aspect_ratio"], 3),
                    }
                )
            self.chart_rows.append({**self.counter.counts, "unknown": self.counter.unknown_count})

        output = draw_overlay(image, detections, tracks, line_y)
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "counts": dict(self.counter.counts),
                "unknown_count": self.counter.unknown_count,
                "unknown_logs": list(self.unknown_logs[-50:]),
                "chart_rows": list(self.chart_rows[-100:]),
            }


def run_browser_camera(detector: ProductDetector, settings: dict) -> None:
    if webrtc_streamer is None or av is None:
        run_browser_snapshot(detector, settings)
        return

    st.info("Mode ini untuk Streamlit via GitHub/Streamlit Cloud. Kamera dibuka dari browser pengguna.")
    ctx = webrtc_streamer(
        key="browser-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: BrowserCameraProcessor(detector, settings, CLASS_NAMES),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("Refresh metrik kamera browser"):
            pass
        state = ctx.video_processor.snapshot()
        render_state_metrics(state["counts"], state["unknown_count"])
        if state["chart_rows"]:
            st.line_chart(pd.DataFrame(state["chart_rows"]))
        if state["unknown_logs"]:
            st.subheader("Log unknown")
            st.dataframe(pd.DataFrame(state["unknown_logs"]), use_container_width=True, hide_index=True)


def run_browser_snapshot(detector: ProductDetector, settings: dict) -> None:
    st.warning(
        "WebRTC tidak tersedia di environment ini. Mode fallback memakai kamera browser bawaan Streamlit "
        "untuk deteksi per snapshot."
    )
    image_file = st.camera_input("Ambil gambar dari kamera browser")
    if image_file is None:
        render_state_metrics({name: 0 for name in CLASS_NAMES}, 0)
        return

    file_bytes = np.frombuffer(image_file.getvalue(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Gambar kamera tidak dapat dibaca.")
        return

    line_y = int(frame.shape[0] * settings["line_position"] / 100)
    detections, unknowns = detector.detect(
        frame,
        confidence_threshold=settings["confidence"],
        min_area=settings["min_area"],
    )
    tracks = [
        type("SnapshotTrack", (), {"centroid": item["centroid"], "track_id": idx + 1})()
        for idx, item in enumerate(detections)
    ]
    output = draw_overlay(frame, detections, tracks, line_y)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    st.image(output, channels="RGB", use_container_width=True)

    counts = {name: 0 for name in CLASS_NAMES}
    unknown_count = 0
    logs: list[dict] = []
    for item in detections:
        if item["class_name"] == "unknown":
            unknown_count += 1
        elif item["class_name"] in counts:
            counts[item["class_name"]] += 1
    for unknown in unknowns:
        logs.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": round(unknown["confidence"], 3),
                "reason": unknown["reason"],
                "area": unknown["area"],
                "aspect_ratio": round(unknown["aspect_ratio"], 3),
            }
        )

    render_state_metrics(counts, unknown_count)
    st.line_chart(pd.DataFrame([{**counts, "unknown": unknown_count}]))
    if logs:
        st.subheader("Log unknown")
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Deteksi Bungkus Minuman", layout="wide")
    init_state()

    st.title("Deteksi dan Counting Bungkus Produk Minuman")
    settings = sidebar()
    input_mode = st.sidebar.radio(
        "Mode input",
        ["Browser camera - GitHub/Streamlit Cloud", "OpenCV camera - local runtime"],
    )

    left, right = st.columns([3, 2])
    video_slot = left.empty()
    metrics_slot = right.empty()
    chart_slot = right.empty()
    log_slot = right.empty()

    controls = st.columns([1, 1, 4])
    if controls[0].button("Start kamera", type="primary"):
        st.session_state.running = True
    if controls[1].button("Stop kamera"):
        st.session_state.running = False
    if controls[2].button("Reset counter"):
        reset_runtime()

    if not CLASS_NAMES:
        st.error("Folder dataset per kelas tidak ditemukan.")
        return

    if not settings["model_path"].exists():
        st.warning(
            "Model belum ditemukan. Jalankan training di Colab, salin best.pt ke models/best.pt, lalu deploy via GitHub/Streamlit Cloud."
        )
        st.code(
            "python -m src.dataset_generator --samples-per-class 200\n"
            "python train_yolo.py --data generated_dataset/dataset.yaml --epochs 80 --batch 8\n"
            "mkdir -p models\n"
            "cp runs/beverage_yolov8n/weights/best.pt models/best.pt\n"
            "# push app.py, src/, requirements.txt, .streamlit/, models/best.pt ke GitHub",
            language="bash",
        )
        render_metrics()
        return

    detector = load_detector(str(settings["model_path"]), tuple(CLASS_NAMES))

    if input_mode == "Browser camera - GitHub/Streamlit Cloud":
        run_browser_camera(detector, settings)
        return

    with metrics_slot.container():
        render_metrics()
    with log_slot.container():
        st.subheader("Log unknown")
        render_logs()

    if not st.session_state.running:
        video_slot.info("Klik Start kamera untuk memulai deteksi real-time.")
        return

    capture = cv2.VideoCapture(settings["camera_index"])
    if not capture.isOpened():
        st.session_state.running = False
        st.error("Kamera tidak dapat dibuka. Cek camera index atau izin kamera.")
        return

    try:
        while st.session_state.running:
            ok, frame = capture.read()
            if not ok:
                st.error("Frame kamera tidak dapat dibaca.")
                break

            height = frame.shape[0]
            line_y = int(height * settings["line_position"] / 100)
            detections, unknowns = detector.detect(
                frame,
                confidence_threshold=settings["confidence"],
                min_area=settings["min_area"],
            )
            tracks = st.session_state.tracker.update(detections)
            st.session_state.counter.apply(tracks, line_y)

            for unknown in unknowns:
                st.session_state.unknown_logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": round(unknown["confidence"], 3),
                        "reason": unknown["reason"],
                        "area": unknown["area"],
                        "aspect_ratio": round(unknown["aspect_ratio"], 3),
                    }
                )

            output = draw_overlay(frame, detections, tracks, line_y)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            video_slot.image(output, channels="RGB", use_container_width=True)

            counter = st.session_state.counter
            row = {**counter.counts, "unknown": counter.unknown_count}
            st.session_state.chart_rows.append(row)
            chart_slot.line_chart(pd.DataFrame(st.session_state.chart_rows[-100:]))

            with metrics_slot.container():
                render_metrics()
            with log_slot.container():
                st.subheader("Log unknown")
                render_logs()

            time.sleep(0.03)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
