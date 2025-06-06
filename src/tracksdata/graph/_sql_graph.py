from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import sqlalchemy as sa
from numpy.typing import ArrayLike
from sqlalchemy.orm import DeclarativeBase, Session, load_only
from sqlalchemy.sql.type_api import TypeEngine

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


def _is_builtin(obj: Any) -> bool:
    return getattr(obj.__class__, "__module__", None) == "builtins"


class Base(DeclarativeBase):
    pass


class Node(Base):
    __tablename__ = "nodes"
    t = sa.Column(sa.Integer, primary_key=True)
    node_id = sa.Column(sa.BigInteger, primary_key=True, unique=True)


class Edge(Base):
    __tablename__ = "edges"
    edge_id = sa.Column(sa.Integer, primary_key=True, unique=True, autoincrement=True)
    source_id = sa.Column(sa.BigInteger, sa.ForeignKey("nodes.node_id"))
    target_id = sa.Column(sa.BigInteger, sa.ForeignKey("nodes.node_id"))


class SQLGraph(BaseGraph):
    node_id_time_multiplier = 1_000_000_000

    def __init__(self, db_path: str):
        self._engine = sa.create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self._engine)

        self._max_id_per_time = {}
        self._update_max_id_per_time()

    def _update_max_id_per_time(self) -> None:
        with Session(self._engine) as session:
            for t in session.query(Node.t).distinct():
                self._max_id_per_time[t] = int(session.query(sa.func.max(Node.node_id)).scalar())

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

        node = Node(
            node_id=node_id,
            **attributes,
        )

        with Session(self._engine) as session:
            session.add(node)
            session.commit()

        self._max_id_per_time[time] = node_id

        return node_id

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

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            **attributes,
        )

        with Session(self._engine) as session:
            session.add(edge)
            session.commit()
            edge_id = edge.edge_id

        return edge_id

    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> np.ndarray:
        with Session(self._engine) as session:
            query = session.query(Node.node_id)
            for key, value in attributes.items():
                query = query.filter(getattr(Node, key) == value)
            return np.asarray([i for (i,) in query.all()], dtype=int)

    def node_ids(self) -> np.ndarray:
        with Session(self._engine) as session:
            return np.asarray(
                [i for (i,) in session.query(Node.node_id).all()],
                dtype=int,
            )

    def subgraph(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        node_attr_filter: dict[str, Any] | None = None,
        edge_attr_filter: dict[str, Any] | None = None,
        node_feature_keys: Sequence[str] | str | None = None,
        edge_feature_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        raise NotImplementedError("SQLGraph does not support subgraphs")

    def time_points(self) -> list[int]:
        with Session(self._engine) as session:
            return [t for (t,) in session.query(Node.t).distinct().all()]

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

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]

        with Session(self._engine) as session:
            query = session.query(Node)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                query = query.filter(Node.node_id.in_(node_ids))

            if feature_keys is not None:
                query = query.options(
                    load_only(
                        *[getattr(Node, key) for key in feature_keys],
                    ),
                )

            LOG.info("Query: %s", query.statement)

        nodes_df = pl.read_database(
            query.statement,
            connection=session.connection(),
        )

        # match node_ids ordering
        if node_ids is not None:
            nodes_df = self._reorder_by_indices(nodes_df, node_ids, "node_id")

        # indices are included by default and must be removed
        return nodes_df.select([pl.col(c) for c in feature_keys])

    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
        include_targets: bool = False,
    ) -> pl.DataFrame:
        with Session(self._engine) as session:
            query = session.query(Edge)

            if node_ids is not None:
                if hasattr(node_ids, "tolist"):
                    node_ids = node_ids.tolist()

                if include_targets:
                    query = query.filter(Edge.source_id.in_(node_ids))
                else:
                    query = query.filter(
                        Edge.source_id.not_in_(node_ids),
                        Edge.target_id.in_(node_ids),
                    )

            if feature_keys is not None:
                query = query.options(
                    load_only(
                        *[getattr(Edge, key) for key in feature_keys],
                    ),
                )

            LOG.info("Query: %s", query.statement)

            return pl.read_database(
                query.statement,
                connection=session.connection(),
            )

    @property
    def node_features_keys(self) -> list[str]:
        keys = list(Node.__table__.columns.keys())
        return keys

    @property
    def edge_features_keys(self) -> list[str]:
        keys = list(Edge.__table__.columns.keys())
        return keys

    def _sqlalchemy_type_inference(self, default_value: Any) -> TypeEngine:
        if isinstance(default_value, np.ndarray) and np.isscalar(default_value):
            default_value = default_value.item()

        if isinstance(default_value, float):
            return sa.Float

        elif isinstance(default_value, int):
            return sa.Integer

        elif isinstance(default_value, str):
            return sa.String

        elif isinstance(default_value, bool):
            return sa.Boolean

        elif isinstance(default_value, Enum):
            return sa.Enum(default_value.__class__)

        elif default_value is None or not _is_builtin(default_value):
            return sa.PickleType

        else:
            raise ValueError(f"Unsupported default value type: {type(default_value)}")

    def _add_new_column(
        self,
        table_class: type[Base],
        key: str,
        default_value: Any,
    ) -> None:
        sa_type = self._sqlalchemy_type_inference(default_value)
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

        self._add_new_column(Node, key, default_value)

    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        if key in self.edge_features_keys:
            raise ValueError(f"Edge feature key {key} already exists")

        self._add_new_column(Edge, key, default_value)

    @property
    def num_edges(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(Edge).count())

    @property
    def num_nodes(self) -> int:
        with Session(self._engine) as session:
            return int(session.query(Node).count())

    def _update_table(
        self,
        table_class: type[Base],
        ids: Sequence[int],
        id_key: str,
        attributes: dict[str, Any],
    ) -> None:
        attributes = attributes.copy()

        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        for k, v in attributes.items():
            if np.isscalar(v):
                if hasattr(v, "item"):
                    v = v.item()
                attributes[k] = [v] * len(ids)
            else:
                if hasattr(v, "tolist"):
                    attributes[k] = v.tolist()

        with Session(self._engine) as session:
            query = session.query(table_class)
            query = query.filter(getattr(table_class, id_key) == sa.bindparam(f"bind_{id_key}"))
            query.update(
                {k: sa.bindparam(f"bind_{k}") for k in attributes},
            )

            # must be done after the query is created
            attributes[id_key] = ids
            LOG.info("update %s table with:\n%s", table_class.__table__, attributes)

            session.execute(
                query,
                {f"bind_{k}": v for k, v in attributes.items()},
            )
            session.commit()

    def update_node_features(
        self,
        *,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        self._update_table(Node, node_ids, "node_id", attributes)

    def update_edge_features(
        self,
        *,
        edge_ids: ArrayLike,
        attributes: dict[str, Any],
    ) -> None:
        self._update_table(Edge, edge_ids, "edge_id", attributes)
