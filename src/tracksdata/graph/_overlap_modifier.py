from typing import Any

import polars as pl
from tqdm import tqdm

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask


def check_node_overlaps(graph: BaseGraph) -> BaseGraph:
    """
    Wraps to a `BaseGraph` object to insert node overlaps when adding nodes.

    Parameters
    ----------
    graph : BaseGraph
        The graph to wrap, modified in place.

    Returns
    -------
    BaseGraph
        The wrapped graph.
    """
    original_add_node = graph.add_node
    original_bulk_add_nodes = graph.bulk_add_nodes

    def _add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        new_node_id = original_add_node(attrs, validate_keys)
        new_mask: Mask = attrs[DEFAULT_ATTR_KEYS.MASK]

        node_ids = graph.filter_nodes_by_attrs(NodeAttr("t") == attrs["t"])
        node_attrs = graph.node_attrs(
            node_ids=node_ids,
            attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK],
        )

        for node_id, node_mask in zip(
            node_attrs[DEFAULT_ATTR_KEYS.NODE_ID], node_attrs[DEFAULT_ATTR_KEYS.MASK], strict=True
        ):
            if new_mask.intersection(node_mask) > 0 and new_node_id != node_id:
                graph.add_overlap(new_node_id, node_id)

        return new_node_id

    def _bulk_add_nodes(self, nodes: list[dict[str, Any]]) -> list[int]:
        new_node_ids = original_bulk_add_nodes(nodes)
        new_nodes_df = pl.DataFrame(
            {
                DEFAULT_ATTR_KEYS.NODE_ID: new_node_ids,
                DEFAULT_ATTR_KEYS.MASK: [nodes[DEFAULT_ATTR_KEYS.MASK] for node in nodes],
                DEFAULT_ATTR_KEYS.T: [node["t"] for node in nodes],
            }
        )

        overlaps = []

        for (t,), nodes_at_t in tqdm(
            list(new_nodes_df.group_by(DEFAULT_ATTR_KEYS.T)),
            desc="Adding nodes overlaps",
        ):
            node_ids = graph.filter_nodes_by_attrs(NodeAttr("t") == t)
            node_attrs = graph.node_attrs(
                node_ids=node_ids,
                attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK],
            )

            for new_node_id, new_mask in zip(
                nodes_at_t[DEFAULT_ATTR_KEYS.NODE_ID], nodes_at_t[DEFAULT_ATTR_KEYS.MASK], strict=True
            ):
                for node_id, node_mask in zip(
                    node_attrs[DEFAULT_ATTR_KEYS.NODE_ID], node_attrs[DEFAULT_ATTR_KEYS.MASK], strict=True
                ):
                    if new_mask.intersection(node_mask) > 0 and new_node_id != node_id:
                        overlaps.append([new_node_id, node_id])

        graph.bulk_add_overlaps(overlaps)

        return new_node_ids

    graph.add_node = _add_node
    graph.bulk_add_nodes = _bulk_add_nodes

    return graph
