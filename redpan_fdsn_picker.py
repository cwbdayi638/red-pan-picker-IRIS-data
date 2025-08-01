#!/usr/bin/env python3
"""Example application using RED-PAN to pick phases from IRIS FDSN data.

The script downloads waveform data for a single earthquake event from the
IRIS FDSN service, runs the RED-PAN picker and prints the resulting phase
picks.  A pretrained RED-PAN model is required.  See the project
instructions for how to obtain the model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

# Third party modules provided by RED-PAN
from redpan.factory import inference_engine
from redpan.picks import extract_picks
from redpan.utils import stream_standardize
from tensorflow.keras.models import load_model


def fetch_event_waveforms(
    event_id: str,
    network: str = "IU",
    station: str = "ANMO",
    location: str = "00",
    channel: str = "BH?",
    pretime: float = 60.0,
    posttime: float = 120.0,
) -> Stream:
    """Download waveform data for an event from IRIS.

    Parameters
    ----------
    event_id:
        Event identifier in the IRIS catalog.
    network, station, location, channel:
        Station selectors passed to :func:`obspy.clients.fdsn.Client.get_waveforms`.
    pretime, posttime:
        Seconds of data before and after the event origin time to request.
    """
    client = Client("IRIS")
    event = client.get_events(eventid=event_id)[0]
    origin = event.preferred_origin() or event.origins[0]
    t0 = origin.time

    st = client.get_waveforms(
        network,
        station,
        location,
        channel,
        t0 - pretime,
        t0 + posttime,
    )

    st.merge(method=1)  # ensure a single trace for each component
    st.detrend("demean")
    st.filter("bandpass", freqmin=1.0, freqmax=20.0)

    # order channels Z, N, E
    st_z = st.select(channel="?HZ")[0]
    st_n = st.select(channel="?HN")[0]
    st_e = st.select(channel="?HE")[0]
    return Stream([st_z, st_n, st_e])


def load_picker(model_path: Path):
    """Load RED-PAN model and build inference engine."""
    model = load_model(model_path, compile=False)
    return inference_engine(model)


def main():
    parser = argparse.ArgumentParser(description="Run RED-PAN picker on IRIS data")
    parser.add_argument("event_id", help="IRIS event identifier")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("REDPAN_60s_240107/train.hdf5"),
        help="Path to pretrained RED-PAN model",
    )
    parser.add_argument("--network", default="IU")
    parser.add_argument("--station", default="ANMO")
    parser.add_argument("--location", default="00")
    parser.add_argument("--channel", default="BH?")
    args = parser.parse_args()

    st = fetch_event_waveforms(
        args.event_id,
        network=args.network,
        station=args.station,
        location=args.location,
        channel=args.channel,
    )

    # Standardize before prediction when data length < window size
    st_std = stream_standardize(st)

    picker = load_picker(args.model)
    p_stream, s_stream, m_stream = picker.annotate_stream(st_std, postprocess=True)

    picks_df = extract_picks(st_std, p_stream, s_stream, m_stream)
    print(picks_df)


if __name__ == "__main__":
    main()
