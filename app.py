import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="CRT Magnetic Field Analyzer",
    page_icon="🧲",
    layout="wide"
)

st.title("🧲 Calibrated CRT Magnetic Field Visualization, Mapping & Attenuation")
st.caption(
    "Segmented analysis of CRT color response. The no-magnet reference is separated from the magnetic-source and object-response measurements."
)

st.warning(
    "An image or video does not contain an absolute mT value by itself. "
    "Use calibration images measured with a Hall probe or gaussmeter."
)


def prep(img, size=(640, 480)):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    side = int(min(w, h) * 0.85)
    x = (w - side) // 2
    y = (h - side) // 2
    crop = arr[y:y + side, x:x + side]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def rgb_values(img):
    a = np.asarray(img).astype(np.float32)
    return a[:, :, 0].mean(), a[:, :, 1].mean(), a[:, :, 2].mean()


def response(reference, test):
    a = prep(reference).astype(np.float32) / 255.0
    b = prep(test).astype(np.float32) / 255.0
    rgb_change = np.mean(np.abs(a - b))

    la = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    color_change = np.mean(np.linalg.norm(la - lb, axis=2)) / 100.0

    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ea = cv2.Canny(ga, 50, 150)
    eb = cv2.Canny(gb, 50, 150)
    edge_change = np.mean(cv2.absdiff(ea, eb)) / 255.0

    return float(0.50 * rgb_change + 0.35 * color_change + 0.15 * edge_change)


def diff_image(a, b):
    x = prep(a)
    y = prep(b)
    d = cv2.absdiff(x, y)
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)


def heatmap(reference, test, n=30):
    a = prep(reference).astype(np.float32) / 255.0
    b = prep(test).astype(np.float32) / 255.0
    d = np.mean(np.abs(a - b), axis=2)
    h, w = d.shape
    out = np.zeros((n, n), dtype=np.float32)
    for r in range(n):
        y1, y2 = int(r * h / n), int((r + 1) * h / n)
        for c in range(n):
            x1, x2 = int(c * w / n), int((c + 1) * w / n)
            cell = d[y1:y2, x1:x2]
            if cell.size:
                out[r, c] = cell.mean()
    return out


def build_model(x, y, degree):
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    Y = np.asarray(y, dtype=float)
    if degree == 1:
        model = LinearRegression()
    else:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, Y)
    return model


def analyze_video(path, reference, every=5, max_frames=300):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    frame_no = 0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no % every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            r, g, b = rgb_values(img)
            rows.append({
                "Frame": frame_no,
                "Time_s": frame_no / fps,
                "Red": r,
                "Green": g,
                "Blue": b,
                "CRT_Response": response(reference, img)
            })
            count += 1
            if count >= max_frames:
                break

        frame_no += 1

    cap.release()
    return pd.DataFrame(rows)


# ---------------- SIDEBAR CALIBRATION ----------------

st.sidebar.header("🧲 Calibration")

cal_ref_file = st.sidebar.file_uploader(
    "Reference / zero-field image",
    type=["png", "jpg", "jpeg", "webp"],
    key="cal_ref"
)

cal_files = st.sidebar.file_uploader(
    "Calibration images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key="cal_files"
)

field_text = st.sidebar.text_input(
    "Known fields in mT",
    placeholder="0,5,10,20,30,40"
)

degree = st.sidebar.selectbox("Calibration curve", [1, 2, 3], index=1)

model = None

if cal_ref_file and cal_files and field_text.strip():
    try:
        fields = [float(x.strip()) for x in field_text.split(",") if x.strip()]
        if len(fields) != len(cal_files):
            st.sidebar.error("Number of field values must equal number of calibration images.")
        elif len(fields) < degree + 1:
            st.sidebar.error("Add more calibration points.")
        else:
            ref = Image.open(cal_ref_file).convert("RGB")
            responses = []
            for f in cal_files:
                responses.append(response(ref, Image.open(f).convert("RGB")))

            table = pd.DataFrame({
                "Image": [f.name for f in cal_files],
                "Field_mT": fields,
                "CRT_Response": responses
            }).sort_values("Field_mT").reset_index(drop=True)

            model = build_model(
                table["CRT_Response"].values,
                table["Field_mT"].values,
                degree
            )
            st.sidebar.success("Calibration loaded.")
            st.sidebar.write(table)

    except Exception as e:
        st.sidebar.error("Calibration error: " + str(e))


# ---------------- TABS ----------------

photo, video = st.tabs(["📷 PHOTO ANALYSIS", "🎥 VIDEO ANALYSIS"])

with photo:
    st.header("Photo analysis")
    c1, c2 = st.columns(2)

    with c1:
        f1 = st.file_uploader(
            "A — Magnet only",
            type=["png", "jpg", "jpeg", "webp"],
            key="photo_a"
        )

    with c2:
        f2 = st.file_uploader(
            "B — Magnet + object",
            type=["png", "jpg", "jpeg", "webp"],
            key="photo_b"
        )

    if f1 and f2:
        a = Image.open(f1).convert("RGB")
        b = Image.open(f2).convert("RGB")
        ref = Image.open(cal_ref_file).convert("RGB") if cal_ref_file else a

        ra = response(ref, a)
        rb = response(ref, b)
        attenuation = max(0.0, min(100.0, (ra - rb) / ra * 100.0)) if ra > 1e-9 else 0.0

        fa = model.predict([[ra]])[0] if model else None
        fb = model.predict([[rb]])[0] if model else None

        x1, x2, x3 = st.columns(3)
        x1.metric("Magnet response", f"{ra:.6f}")
        x2.metric("Object response", f"{rb:.6f}")
        x3.metric("Response reduction", f"{attenuation:.2f}%")

        if model:
            y1, y2 = st.columns(2)
            y1.metric("Estimated field — magnet", f"{float(fa):.3f} mT")
            y2.metric("Estimated field — object", f"{float(fb):.3f} mT")

        i1, i2, i3 = st.columns(3)
        i1.image(a, caption="Magnet only", use_container_width=True)
        i2.image(b, caption="Magnet + object", use_container_width=True)
        i3.image(diff_image(a, b), caption="Difference", use_container_width=True)

        st.subheader("CRT response heatmap")
        st.image(heatmap(ref, a), caption="Relative CRT response", use_container_width=True)

with video:
    st.header("🎥 Video magnetic-field analysis")
    st.write("This mode follows the changing red/green/blue CRT response frame by frame.")

    vf = st.file_uploader(
        "Upload CRT video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video"
    )

    vr = st.file_uploader(
        "Reference CRT image",
        type=["png", "jpg", "jpeg", "webp"],
        key="video_ref"
    )

    every = st.slider("Analyze every Nth frame", 1, 30, 5)
    max_frames = st.slider("Maximum frames", 50, 1000, 300, 50)

    if vf and vr and st.button("🧲 ANALYZE VIDEO", type="primary"):
        reference = Image.open(vr).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vf.read())
        tmp.close()

        with st.spinner("Analyzing video..."):
            data = analyze_video(tmp.name, reference, every, max_frames)

        st.success("Video analysis completed.")

        st.subheader("🔴🟢🔵 Color response")
        st.line_chart(data[["Time_s", "Red", "Green", "Blue"]].set_index("Time_s"))

        st.subheader("🧲 CRT magnetic response")
        st.line_chart(data[["Time_s", "CRT_Response"]].set_index("Time_s"))

        if model:
            data["Estimated_Field_mT"] = model.predict(data[["CRT_Response"]])
            st.subheader("Estimated magnetic field")
            st.line_chart(data[["Time_s", "Estimated_Field_mT"]].set_index("Time_s"))
            st.metric(
                "Maximum estimated field",
                f"{data['Estimated_Field_mT'].max():.3f} mT"
            )
        else:
            st.info("Load calibration images to convert response into estimated mT.")

        st.dataframe(data, use_container_width=True)
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download video analysis CSV",
            csv,
            "crt_magnetic_field_video.csv",
            "text/csv"
        )

        try:
            os.remove(tmp.name)
        except Exception:
            pass

st.divider()
st.caption(
    "Experimental image-analysis prototype. Validate mT values with a calibrated Hall probe or gaussmeter."
)
