# CRT Magnetic Field Analyzer

Segmented Streamlit application for experimental CRT-based magnetic-field visualization, mapping, calibration and attenuation analysis.

## Features

- Magnet/magnetic-source CRT image analysis without requiring a separate no-magnet image.
- RGB response measurement.
- Difference-image visualization.
- CRT response heatmap.
- Calibration against independently measured magnetic-field values in mT.
- Linear, quadratic and cubic calibration curves.
- Magnet-only versus magnet-plus-object attenuation comparison.
- Frame-by-frame video analysis.
- CSV export of video measurements.
- Optional estimated magnetic-field output after calibration.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Create/select a GitHub repository.
2. Upload `app.py`, `requirements.txt` and `README.md`.
3. Deploy `app.py` as the main file.

## Calibration note

The CRT response calculated from an image is a relative experimental measurement. Absolute magnetic-field values in mT require calibration using images acquired under independently measured magnetic fields, preferably with a calibrated Hall probe or gaussmeter.

## Experimental use

For reproducible measurements, keep camera position, exposure, focus, CRT brightness, magnet position and image-processing conditions consistent between calibration and test measurements.
