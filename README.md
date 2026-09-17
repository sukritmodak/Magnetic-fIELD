# CRT Magnetic Field Analyzer

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   pip install -r requirements.txt
4. Then:
   streamlit run app.py

## Important
The app cannot determine absolute magnetic field strength from an arbitrary photograph.
It needs calibration images taken on the same CRT/setup with independently measured
magnetic-field values (for example from a Hall probe/gaussmeter).

The attenuation result is an image-response reduction metric. Validate it independently
before treating it as a physical shielding/attenuation measurement.
