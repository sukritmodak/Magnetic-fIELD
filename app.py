import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="CRT Magnetic Field Analyzer", layout="wide")

st.title("CRT Magnetic Field Analyzer")
st.caption("Prototype: image-based estimation from a calibrated CRT response.")

st.warning(
    "An ordinary photo cannot determine magnetic field strength in mT/G by itself. "
    "This app estimates field strength only after calibration with known magnetic-field "
    "values measured at the same CRT/setup. Treat the result as an experimental estimate, "
    "not as a calibrated gaussmeter."
)

def load_image(uploaded):
    return Image.open(uploaded).convert("RGB")

def center_crop(im, frac=0.80):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch))

def prep(im, size=(640, 480)):
    arr = np.array(im)
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)

def response_score(reference, test):
    """
    Dimensionless image-response metric.

    It combines:
      - luminance/color change
      - RGB pixel change
      - edge/geometry change

    This is a calibration metric, NOT a direct physical magnetic-field unit.
    """
    ref = prep(center_crop(reference))
    tst = prep(center_crop(test))

    ref_f = ref.astype(np.float32) / 255.0
    tst_f = tst.astype(np.float32) / 255.0

    lab_r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_t = cv2.cvtColor(tst, cv2.COLOR_RGB2LAB).astype(np.float32)
    color_delta = np.mean(np.linalg.norm(lab_t - lab_r, axis=2)) / 100.0

    gray_r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    gray_t = cv2.cvtColor(tst, cv2.COLOR_RGB2GRAY)
    edges_r = cv2.Canny(gray_r, 50, 150)
    edges_t = cv2.Canny(gray_t, 50, 150)
    edge_delta = np.mean(cv2.absdiff(edges_r, edges_t)) / 255.0

    pixel_delta = np.mean(np.abs(tst_f - ref_f))

    return float(
        0.55 * color_delta +
        0.30 * pixel_delta +
        0.15 * edge_delta
    )

def fit_calibration(x, y, degree=2):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    if degree == 1:
        return LinearRegression().fit(x, y)

    return make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    ).fit(x, y)

def r2_score_manual(model, x, y):
    pred = model.predict(np.asarray(x).reshape(-1, 1))
    y = np.asarray(y)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot

def estimate_uncertainty(model, x, y, x_new):
    pred = model.predict(np.asarray(x).reshape(-1, 1))
    residual = np.asarray(y) - pred

    if len(residual) < 3:
        return None

    rmse = float(np.sqrt(np.mean(residual ** 2)))

    eps = max(abs(float(x_new)) * 1e-4, 1e-6)
    y1 = float(model.predict([[float(x_new) - eps]])[0])
    y2 = float(model.predict([[float(x_new) + eps]])[0])
    slope = abs((y2 - y1) / (2 * eps))

    return rmse * slope if slope > 1e-12 else None


# ---------------- Calibration ----------------
st.sidebar.header("1. Calibration")
st.sidebar.write(
    "Calibration must use the same CRT, camera, distance, exposure, brightness, "
    "contrast and magnet geometry used during measurement."
)

ref_cal_file = st.sidebar.file_uploader(
    "Calibration reference image (usually zero-field)",
    type=["png", "jpg", "jpeg", "webp"],
    key="ref_cal"
)

cal_files = st.sidebar.file_uploader(
    "Calibration test images (multiple)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key="cal_files"
)

known_text = st.sidebar.text_input(
    "Known field values in mT, in the SAME order as calibration files",
    placeholder="Example: 5, 10, 20, 40"
)

degree = st.sidebar.selectbox(
    "Calibration curve",
    [1, 2, 3],
    index=1
)

model = None
cal_table = None

if ref_cal_file and cal_files and known_text.strip():
    try:
        known = [
            float(v.strip())
            for v in known_text.split(",")
            if v.strip()
        ]

        if len(known) != len(cal_files):
            st.sidebar.error(
                "The number of field values must equal the number of calibration images."
            )
        elif len(cal_files) < degree + 1:
            st.sidebar.error(
                f"Use at least {degree + 1} calibration images for a degree-{degree} curve."
            )
        else:
            ref_cal = load_image(ref_cal_file)

            scores = []
            for f in cal_files:
                scores.append(
                    response_score(ref_cal, load_image(f))
                )

            cal_table = pd.DataFrame({
                "Image": [f.name for f in cal_files],
                "Known field (mT)": known,
                "Response score": scores
            }).sort_values("Known field (mT)")

            model = fit_calibration(
                cal_table["Response score"].values,
                cal_table["Known field (mT)"].values,
                degree=degree
            )

            r2 = r2_score_manual(
                model,
                cal_table["Response score"].values,
                cal_table["Known field (mT)"].values
            )

            st.sidebar.success(
                f"Calibration loaded • R² = {r2:.4f}"
            )

    except Exception as e:
        st.sidebar.error(f"Calibration error: {e}")


# ---------------- Measurement ----------------
st.header("2. Measurement")

c1, c2 = st.columns(2)

with c1:
    ref_file = st.file_uploader(
        "A — Magnet present, NO object between magnet and CRT",
        type=["png", "jpg", "jpeg", "webp"],
        key="measurement_ref"
    )

with c2:
    obj_file = st.file_uploader(
        "B — Same magnet/setup, object inserted between magnet and CRT",
        type=["png", "jpg", "jpeg", "webp"],
        key="measurement_obj"
    )

if ref_file and obj_file:
    ref = load_image(ref_file)
    obj = load_image(obj_file)

    if ref_cal_file:
        calibration_reference = load_image(ref_cal_file)
    else:
        calibration_reference = ref

    s_ref = response_score(calibration_reference, ref)
    s_obj = response_score(calibration_reference, obj)

    attenuation = None
    if abs(s_ref) > 1e-12:
        attenuation = max(
            0.0,
            min(100.0, (s_ref - s_obj) / s_ref * 100.0)
        )

    est_ref = None
    est_obj = None
    unc_ref = None
    unc_obj = None

    if model is not None:
        est_ref = float(model.predict([[s_ref]])[0])
        est_obj = float(model.predict([[s_obj]])[0])

        unc_ref = estimate_uncertainty(
            model,
            cal_table["Response score"],
            cal_table["Known field (mT)"],
            s_ref
        )

        unc_obj = estimate_uncertainty(
            model,
            cal_table["Response score"],
            cal_table["Known field (mT)"],
            s_obj
        )

    st.subheader("Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Response — magnet only", f"{s_ref:.6f}")
        if est_ref is not None:
            st.metric(
                "Estimated field — magnet only",
                f"{est_ref:.3f} mT"
            )

    with r2:
        st.metric("Response — object inserted", f"{s_obj:.6f}")
        if est_obj is not None:
            st.metric(
                "Estimated field — after object",
                f"{est_obj:.3f} mT"
            )

    with r3:
        if attenuation is not None:
            st.metric(
                "Estimated attenuation",
                f"{attenuation:.2f}%"
            )
        else:
            st.metric("Estimated attenuation", "—")

    if unc_ref is not None or unc_obj is not None:
        st.caption(
            "Approximate calibration-model uncertainty only: "
            f"before ≈ {unc_ref:.3f} mT, after ≈ {unc_obj:.3f} mT. "
            "Camera noise, CRT drift and geometry errors are not included."
        )

    st.subheader("CRT images")

    i1, i2, i3 = st.columns(3)

    i1.image(
        ref,
        caption="A — Magnet only",
        use_container_width=True
    )

    i2.image(
        obj,
        caption="B — Object inserted",
        use_container_width=True
    )

    a = prep(center_crop(ref))
    b = prep(center_crop(obj))

    diff = cv2.absdiff(a, b)
    diff = cv2.normalize(
        diff, None, 0, 255, cv2.NORM_MINMAX
    )

    i3.image(
        diff,
        caption="A vs B difference",
        use_container_width=True
    )

    st.subheader("Interpretation")

    if model is None:
        st.info(
            "No absolute mT estimate is shown because no calibration curve is loaded. "
            "Upload calibration images with known field values in the sidebar."
        )
    else:
        st.write(
            f"The calibrated model maps the image-response score to approximately "
            f"{est_ref:.3f} mT before the object and "
            f"{est_obj:.3f} mT after the object."
        )

    st.write(
        "The attenuation number is the percentage reduction in the image-response "
        "metric between the magnet-only image and the object-inserted image. "
        "It is not automatically a material shielding coefficient. "
        "For a physical attenuation claim, independently validate the field before "
        "and after the object with a Hall probe/gaussmeter."
    )

else:
    st.info(
        "Upload the two measurement images above. "
        "For an absolute mT estimate, first create a calibration set "
        "using known magnetic-field measurements."
    )


# ---------------- Experimental procedure ----------------
st.divider()
st.header("Recommended calibration procedure")

st.markdown(
    """
**For a useful experimental result:**

1. Keep the CRT, camera position, focus, exposure, brightness and contrast fixed.
2. Place a calibrated Hall probe/gaussmeter at the same measurement location.
3. Record several known fields, for example 0, 5, 10, 20 and 30 mT.
4. Capture one CRT image at each known field.
5. Upload those images as the calibration set and enter their measured mT values.
6. For attenuation, capture **magnet only** and then **magnet + test object**.
7. Do not move the camera, CRT or magnet between those two measurements.
8. Repeat every point several times and average the results.
"""
)

st.caption(
    "Prototype image-analysis software. Absolute magnetic-field accuracy depends "
    "on the calibration data and the physical CRT/electron-beam setup."
)
