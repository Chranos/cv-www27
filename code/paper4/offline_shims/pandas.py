"""Offline shim, NOT pandas.

The official BOLT select_frames.py imports pandas at module top (its dataset
parquet loaders use it: get_query_list / read_parquet). Those loaders are not used
by the exploratory arm -- the arm reads the selection cell's own registered
110-question pool from the on-device LongVideoBench val copy (mvp_common.load_pool),
so no parquet is ever read. This module exists only so the byte-exact vendored file
imports; any actual use fails loudly.
"""


def read_parquet(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "pandas is not installed on this device and the exploratory BOLT arm does not "
        "use the official parquet dataset loaders; if you see this, a code path that "
        "was declared unused was reached")


def read_csv(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("see read_parquet")
