from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rustworkx as rx
import sqlalchemy as sa
from numpy.typing import ArrayLike
from sqlalchemy.orm import DeclarativeBase, Session, load_only
from sqlalchemy.sql.type_api import TypeEngine

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._dataframe import unpack_array_features
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


def _is_builtin(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", None) == "builtins"


def _data_numpy_to_native(data: dict[str, Any]) -> None:
    """
    Convert numpy scalars to native Python scalars in place.

    Parameters
    ----------
    data : dict[str, Any]
        The data to convert.
    """
    for k, v in data.items():
        if np.isscalar(v) and hasattr(v, "item"):
            data[k] = v.item()


class SQLGraph(BaseGraph):
    node_id_time_multiplier: int = 1_000_000_000
    Base: type[DeclarativeBase]
    Node: type[DeclarativeBase]
    Edge: type[DeclarativeBase]

    def __init__(
        self,
        drivername: str,
        database: str,
        username: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
        overwrite: bool = False,
    ):
        # Create unique classes for this instance
        self._define_schema()

        self._url = sa.engine.URL.create(
            drivername,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self._engine = sa.create_engine(self._url)

        if overwrite:
            self.Base.metadata.drop_all(self._engine)

        self.Base.metadata.create_all(self._engine)

        self._max_id_per_time = {}
        self._update_max_id_per_time()

    def _define_schema(self) -> None:
        """Define the database schema classes for this SQLGraph instance."""

        class Base(DeclarativeBase):
            pass

        class Node(Base):
            __tablename__ = "nodes"

            # Use node_id as sole primary key for simpler updates
            node_id = sa.Column(sa.BigInteger, primary_key=True, unique=True)

            # Add t as a regular column
            # NOTE might want to use as index for fast time-based queries
            t = sa.Column(sa.Integer, nullable=False)

        class Edge(Base):
            __tablename__ = "edges"
            edge_id = sa.Column(sa.Integer, primary_key=True, unique=True, autoincrement=True)
            source_id = sa.Column(sa.BigInteger, sa.ForeignKey("nodes.node_id"))
            target_id = sa.Column(sa.BigInteger, sa.ForeignKey("nodes.node_id"))

        # Assign to instance variables
        self.Base = Base
        self.Node = Node
        self.Edge = Edge

        self._boolean_columns = {self.Node: [], self.Edge: []}

    def _cast_boolean_columns(self, table_class: type[DeclarativeBase], df: pl.DataFrame) -> pl.DataFrame:
        """
        This is required because polars bypasses the boolean type and converts it to integer.
        """
        for col in self._boolean_columns[table_class]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Boolean))
        return df

    def _update_max_id_per_time(self) -> None:
        with Session(self._engine) as session:
            for t in session.query(self.Node.t).distinct():
                self._max_id_per_time[t] = int(session.query(sa.func.max(self.Node.node_id)).scalar())

    def add_node(
        self,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        if validate_keys:
            self._validate_attributes(attributes, self.node_features_keys, "node")

            if "t" not in attributes:
                raise ValueError(f"Node attributes must have a 't' key. Got {attributes.keys()}")

        time = attributes["t"]
        default_node_id = (time * self.node_id_time_multiplier) - 1
        node_id = self._max_id_per_time.get(time, default_node_id) + 1

        node = self.Node(
            node_id=node_id,
            **attributes,
        )

        with Session(self._engine) as session:
            session.add(node)
            session.commit()

        self._max_id_per_time[time] = node_id

        return node_id

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> None:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        """
        for node in nodes:
            time = node["t"]
            default_node_id = (time * self.node_id_time_multiplier) - 1
            node_id = self._max_id_per_time.get(time, default_node_id) + 1
            node[DEFAULT_ATTR_KEYS.NODE_ID] = node_id
            self._max_id_per_time[time] = node_id

        with Session(self._engine) as session:
            session.bulk_insert_mappings(self.Node, nodes)
            session.commit()

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        if validate_keys:
            self._validate_attributes(attributes, self.edge_features_keys, "edge")

        if hasattr(source_id, "item"):
            source_id = source_id.item()

        if hasattr(target_id, "item"):
            target_id = target_id.item()

        edge = self.Edge(
            source_id=source_id,
            target_id=target_id,
            **attributes,
        )

        with Session(self._engine) as session:
            session.add(edge)
            session.commit()
            edge_id = edge.edge_id

        return edge_id

    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
    ) -> None:
        """
        Faster method to add multiple edges to the graph with less overhead and fewer checks.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            The data of the edges to be added.
            The keys of the data will be used as the attributes of the edges.
            Must have "source_id" and "target_id" keys.
        """
        for edge in edges:
            _data_numpy_to_native(edge)

        with Session(self._engine) as session:
            session.bulk_insert_mappings(self.Edge, edges)
            session.commit()

    def _get_neighbors(
        self,
        node_key: str,
        neighbor_key: str,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors or sucessors of a list of nodes.
        See more information below.
        """
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]

        with Session(self._engine) as session:
            if feature_keys is None:
                # all columns
                node_columns = [self.Node]
            else:
                node_columns = [getattr(self.Node, key) for key in feature_keys]

            query = session.query(getattr(self.Edge, node_key), *node_columns)
            query = query.join(self.Edge, getattr(self.Edge, neighbor_key) == self.Node.node_id)
            query = query.filter(getattr(self.Edge, node_key).in_(node_ids))

            node_df = pl.read_database(
                query.statement,
                connection=session.connection(),
            )
            node_df = self._cast_boolean_columns(self.Node, node_df)

        if single_node:
            return node_df

        neighbors_dict = {node_id: group for (node_id,), group in node_df.group_by(node_key)}
        for node_id in node_ids:
            if node_id not in neighbors_dict:
                neighbors_dict[node_id] = pl.DataFrame(schema=node_df.schema)

        return neighbors_dict

    def sucessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            node_ids=node_ids,
            feature_keys=feature_keys,
        )

    def predecessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        return self._get_neighbors(
            node_key=DEFAULT_ATTR_KEYS.EDGE_TARGET,
            neighbor_key=DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            node_ids=node_ids,
            feature_keys=feature_keys,
        )

    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> list[int]:
        with Session(self._engine) as session:
            query = session.query(self.Node.node_id)
            for key, value in attributes.items():
                query = query.filter(getattr(self.Node, key) == value)
            return [i for (i,) in query.all()]

    def node_ids(self) -> list[int]:
        with Session(self._engine) as session:
            return [i for (i,) in session.query(self.Node.node_id).all()]

    def subgraph(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        node_attr_filter: dict[str, Any] | None = None,
        edge_attr_filter: dict[str, Any] | None = None,
        node_feature_keys: Sequence[str] | str | None = None,
        edge_feature_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        self._validate_subgraph_args(node_ids, node_attr_filter, edge_attr_filter)

        if hasattr(node_ids, "tolist"):
            node_ids = node_ids.tolist()

        with Session(self._engine) as session:
            # selecting edges
            edge_query = session.query(self.Edge)

            edge_filtered = False
            if edge_attr_filter is not None:
                edge_query = edge_query.filter(
                    *[getattr(self.Edge, k) == v for k, v in edge_attr_filter.items()],
                )

                assert node_ids is None, "node_ids must be None when edge_attr_filter is not None"

                node_ids = edge_query.with_entities(self.Edge.source_id, self.Edge.target_id).all()
                node_ids = np.unique(node_ids).tolist()
                edge_filtered = True

            node_query = session.query(self.Node)

            if node_ids is not None:
                node_query = node_query.filter(self.Node.node_id.in_(node_ids))

            if node_attr_filter is not None:
                node_query = node_query.filter(
                    *[getattr(self.Node, k) == v for k, v in node_attr_filter.items()],
                )
                node_ids = [i for (i,) in node_query.with_entities(self.Node.node_id).all()]

            if not edge_filtered and node_ids is not None:
                # TODO could this be done at the individual node filtering levels?
                edge_query = edge_query.filter(
                    self.Edge.source_id.in_(node_ids),
                    self.Edge.target_id.in_(node_ids),
                )

            if node_feature_keys is not None:
                if DEFAULT_ATTR_KEYS.NODE_ID not in node_feature_keys:
                    node_feature_keys.append(DEFAULT_ATTR_KEYS.NODE_ID)

                node_query = node_query.options(
                    load_only(
                        *[getattr(self.Node, key) for key in node_feature_keys],
                    ),
                )

            if edge_feature_keys is not None:
                edge_feature_keys = set(edge_feature_keys)
                # we always return the source and target id by default
                edge_feature_keys.add(DEFAULT_ATTR_KEYS.EDGE_ID)
                edge_feature_keys.add(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
                edge_feature_keys.add(DEFAULT_ATTR_KEYS.EDGE_TARGET)
                edge_feature_keys = list(edge_feature_keys)

                edge_query = edge_query.options(
                    load_only(
                        *[getattr(self.Edge, key) for key in edge_feature_keys],
                    ),
                )

        node_map_to_root = {}
        node_map_from_root = {}
        rx_graph = rx.PyDiGraph()

        for row in node_query.all():
            data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
            root_node_id = data.pop(DEFAULT_ATTR_KEYS.NODE_ID)
            node_id = rx_graph.add_node(data)
            node_map_to_root[node_id] = root_node_id
            node_map_from_root[root_node_id] = node_id

        for row in edge_query.all():
            data = {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
            source_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)]
            target_id = node_map_from_root[data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)]
            rx_graph.add_edge(source_id, target_id, data)

        graph = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=node_map_to_root,
            root=self,
        )

        return graph

    def time_points(self) -> list[int]:
        with Session(self._engine) as session:
            return [t for (t,) in session.query(self.Node.t).distinct().all()]

    def _reorder_by_indices(
        self,
        df: pl.DataFrame,
        indices: Sequence[int],
        id_key: str,
    ) -> pl.DataFrame:
        # NOTE: maybe we should avoid doing this and return the indices
        order_df = pl.DataFrame(
            {id_key: indices, "order": np.arange(0, len(indices))},
        )
        return order_df.join(df, on=id_key, how="left").drop("order")

    def _raw_query(self, query: sa.sql.Select) -> str:
        # for some reason, the query.statement is not working with polars
        return str(query.statement.compile(dialect=self._engine.dialect, compile_kwargs={"literal_binds": True}))

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]

        with Session(self._engine) as session:
            query = session.query(self.Node)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                query = query.filter(self.Node.node_id.in_(node_ids))

            if feature_keys is not None:
                # making them unique
                feature_keys = list(set(feature_keys))

                query = query.options(
                    load_only(
                        *[getattr(self.Node, key) for key in feature_keys],
                    ),
                )

            LOG.info("Query: %s", query.statement)

        nodes_df = pl.read_database(
            query.statement,
            connection=session.connection(),
        )
        nodes_df = self._cast_boolean_columns(self.Node, nodes_df)

        # match node_ids ordering
        if node_ids is not None and not nodes_df.is_empty():
            nodes_df = self._reorder_by_indices(nodes_df, node_ids, "node_id")

        # indices are included by default and must be removed
        if feature_keys is not None:
            nodes_df = nodes_df.select([pl.col(c) for c in feature_keys])

        if unpack:
            nodes_df = unpack_array_features(nodes_df)

        return nodes_df

    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
        include_targets: bool = False,
        unpack: bool = False,
    ) -> pl.DataFrame:
        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]

        with Session(self._engine) as session:
            query = session.query(self.Edge)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                if include_targets:
                    query = query.filter(self.Edge.source_id.in_(node_ids))
                else:
                    query = query.filter(
                        self.Edge.source_id.in_(node_ids),
                        self.Edge.target_id.in_(node_ids),
                    )

            if feature_keys is not None:
                feature_keys = set(feature_keys)
                # we always return the source and target id by default
                feature_keys.add(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
                feature_keys.add(DEFAULT_ATTR_KEYS.EDGE_TARGET)
                feature_keys = list(feature_keys)

                LOG.info("Edge feature keys: %s", feature_keys)

                query = query.options(
                    load_only(
                        *[getattr(self.Edge, key) for key in feature_keys],
                    ),
                )

            LOG.info("Query: %s", query.statement)
            # for some reason, the query.statement is not working with polars
            edges_df = pl.read_database(
                self._raw_query(query),
                connection=session.connection(),
            )
            edges_df = self._cast_boolean_columns(self.Edge, edges_df)

        if unpack:
            edges_df = unpack_array_features(edges_df)

        return edges_df

    @property
    def node_features_keys(self) -> list[str]:
        keys = list(self.Node.__table__.columns.keys())
        keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)
        return keys

    @property
    def edge_features_keys(self) -> list[str]:
        keys = list(self.Edge.__table__.columns.keys())
        for k in [DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
            keys.remove(k)
        return keys

    def _sqlalchemy_type_inference(self, default_value: Any) -> TypeEngine:
        if np.isscalar(default_value) and hasattr(default_value, "item"):
            default_value = default_value.item()

        if isinstance(default_value, float):
            return sa.Float

        # must come before integer, otherwise it will be interpreted as it
        elif isinstance(default_value, bool):
            return sa.Boolean

        elif isinstance(default_value, int):
            return sa.Integer

        elif isinstance(default_value, str):
            return sa.String

        elif isinstance(default_value, Enum):
            return sa.Enum(default_value.__class__)

        elif default_value is None or not _is_builtin(default_value):
            return sa.PickleType

        else:
            raise ValueError(f"Unsupported default value type: {type(default_value)}")

    def _add_new_column(
        self,
        table_class: type[DeclarativeBase],
        key: str,
        default_value: Any,
    ) -> None:
        sa_type = self._sqlalchemy_type_inference(default_value)

        if sa_type == sa.Boolean:
            self._boolean_columns[table_class].append(key)

        sa_column = sa.Column(key, sa_type, default=default_value)

        str_dialect_type = sa_column.type.compile(dialect=self._engine.dialect)

        add_column_stmt = sa.DDL(
            f"ALTER TABLE {table_class.__table__} ADD "
            f"COLUMN {sa_column.name} {str_dialect_type} "
            f"DEFAULT {default_value}",
        )
        LOG.info("add %s column statement:\n'%s'", table_class.__table__, add_column_stmt)

        # create the new column in the database
        with Session(self._engine) as session:
            session.execute(add_column_stmt)
            session.commit()

        # register the new column in the Node class
        setattr(table_class, key, sa_column)
        table_class.__table__.append_column(sa_column)

    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        if key in self.node_features_keys:
            raise ValueError(f"Node feature key {key} already exists")

        self._add_new_column(self.Node, key, default_value)

    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        if key in self.edge_features_keys:
            raise ValueError(f"Edge feature key {key} already exists")

        self._add_new_column(self.Edge, key, default_value)

    @property
    def num_edges(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Edge).count())

    @property
    def num_nodes(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(self.Node).count())

    def _update_table(
        self,
        table_class: type[DeclarativeBase],
        ids: Sequence[int],
        id_key: str,
        attributes: dict[str, Any],
    ) -> None:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        # Handle array values with bulk_update_mappings
        attributes = attributes.copy()
        _data_numpy_to_native(attributes)

        # specialized case for scalar values - use simple bulk update
        if all(np.isscalar(v) for v in attributes.values()):
            LOG.info("update %s table with scalar values: %s", table_class.__table__, attributes)

            with Session(self._engine) as session:
                session.query(table_class).filter(getattr(table_class, id_key).in_(ids)).update(attributes)
                session.commit()

            return

        # Prepare values for bulk update
        update_data = []

        for k, v in attributes.items():
            if np.isscalar(v):
                # Convert scalar to list of same length as ids
                attributes[k] = [v] * len(ids)
            else:
                if hasattr(v, "tolist"):
                    attributes[k] = v.tolist()

                # Validate length matches ids
                if len(attributes[k]) != len(ids):
                    raise ValueError(f"Length mismatch: {len(attributes[k])} values for {len(ids)} {id_key}s")

        # Create list of dictionaries for bulk update (simple primary key)
        for i, row_id in enumerate(ids):
            row_data = {id_key: row_id}

            # Add the attribute updates
            for k, v_list in attributes.items():
                row_data[k] = v_list[i]
            update_data.append(row_data)

        LOG.info("update %s table with %d rows", table_class.__table__, len(update_data))
        LOG.info("update data sample: %s", update_data[:2])

        with Session(self._engine) as session:
            # Use bulk_update_mappings for efficient bulk updates
            session.bulk_update_mappings(table_class, update_data)
            session.commit()

    def update_node_features(
        self,
        *,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        if "t" in attributes:
            raise ValueError("Node attribute 't' cannot be updated.")

        self._update_table(self.Node, node_ids, DEFAULT_ATTR_KEYS.NODE_ID, attributes)

    def update_edge_features(
        self,
        *,
        edge_ids: ArrayLike,
        attributes: dict[str, Any],
    ) -> None:
        self._update_table(self.Edge, edge_ids, DEFAULT_ATTR_KEYS.EDGE_ID, attributes)

    def _get_degree(
        self,
        node_ids: list[int] | int | None,
        node_key: str,
    ) -> list[int] | int:
        if isinstance(node_ids, int):
            with Session(self._engine) as session:
                query = (
                    session.query(
                        getattr(self.Edge, node_key),
                    )
                    .filter(getattr(self.Edge, node_key) == node_ids)
                    .count()
                )
            return int(query)

        with Session(self._engine) as session:
            # get the number of edges for each using group by and count
            node_id_col = getattr(self.Edge, node_key)
            query = session.query(node_id_col, sa.func.count(node_id_col)).group_by(node_id_col)
            if node_ids is not None:
                query = query.filter(node_id_col.in_(node_ids))
            degree = dict(query.all())

        if node_ids is None:
            # this is necessary to make sure it's the same order as node_ids
            return [degree.get(node_id, 0) for node_id in self.node_ids()]

        return [degree.get(node_id, 0) for node_id in node_ids]

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in-degree of a list of nodes.
        """
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_TARGET)

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """
        return self._get_degree(node_ids, DEFAULT_ATTR_KEYS.EDGE_SOURCE)
