"""Stub: the offline M3ED tracking dataset loader was not included in the drone-space
deployment bundle (DEPLOY_CELL_DRONESPACE.md) -- it is only used by track.py's offline
`track()` CLI, never by the live pipeline (which builds its own inputs from the EVK4
stream + mocap pose + fused_clean_mocap.ply). Exists only so `from track import
build_model` succeeds without the full offline-eval tree."""


class TrackDataset:
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "TrackDataset is not part of the drone-space live deployment bundle; "
            "see DEPLOY_CELL_DRONESPACE.md")
