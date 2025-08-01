# red-pan-picker-IRIS-data

## RED-PAN FDSN example

`redpan_fdsn_picker.py` demonstrates how to use the [RED-PAN](https://github.com/tso1257771/RED-PAN)
phase picker with waveform data retrieved from the IRIS FDSN service.
The script downloads data for a selected event, runs the picker and prints
P and S phase picks.

### Usage

```
python redpan_fdsn_picker.py EVENT_ID --model PATH_TO_MODEL
```

Install dependencies with:

```
pip install obspy tensorflow git+https://github.com/tso1257771/RED-PAN.git
```

A pretrained RED-PAN model (e.g. `REDPAN_60s_240107/train.hdf5`) must be
provided separately.
