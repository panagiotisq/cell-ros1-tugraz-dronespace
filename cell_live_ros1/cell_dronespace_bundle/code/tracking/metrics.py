"""Stub: see track_dataset.py -- the offline metrics/report code was not included in the
drone-space deployment bundle and is never called by the live pipeline."""


def compute_metrics(*a, **k):
    raise NotImplementedError("offline-only; not part of the live deployment bundle")


def format_report(*a, **k):
    raise NotImplementedError("offline-only; not part of the live deployment bundle")
