"""RQ2c undirected-view summaries."""

from __future__ import annotations

import pandas as pd

from .network_views import directed_to_undirected_edges


def summarize_undirected_view(edges: pd.DataFrame) -> pd.DataFrame:
    """Build the RQ2c undirected edge table from directed direct-reference edges."""

    return directed_to_undirected_edges(edges, drop_self_loop=True)
