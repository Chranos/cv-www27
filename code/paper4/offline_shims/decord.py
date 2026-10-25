"""Offline shim, NOT a decoder.

The official BOLT extract_feature.py imports decord at module top (its __main__
block and the fps=1 feature-extraction pipeline use it). This device has no decord
and nothing may be installed at run time, and the exploratory arm does not use any
decord code path: it scores the cell's registered 32-frame candidate pool decoded by
the cell's cv2 path (registered deviation in plan-exploratory-bolt-timing
setup sec.3). This module exists only so the byte-exact vendored file imports; any
actual use fails loudly instead of silently substituting another decoder.
"""


class VideoReader:  # pragma: no cover - never used, fails loudly if ever reached
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "decord is not installed on this device and the exploratory BOLT arm does "
            "not use the official fps=1 decord pipeline; if you see this, a code path "
            "that was declared unused was reached")
