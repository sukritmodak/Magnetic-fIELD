# CRT Magnetic Field Analyzer — Fixed Version

This version fixes the Streamlit Cloud `NameError`:

```text
NameError: name 'tempfile' is not defined
```

The fix is the explicit import:

```python
import tempfile
```

at the top of `app.py`.

## Main features

- No-magnet image is NOT required for normal photo analysis.
- Magnet/magnetic-source CRT image is the primary input.
- Optional magnet + test-object image for attenuation comparison.
- Configurable CRT segmentation from 2x2 through 8x8.
