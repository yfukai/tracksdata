from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import sqlalchemy as sa
from numpy.typing import ArrayLike
from sqlalchemy.orm import DeclarativeBase, Session
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
    edge_id = sa.Column(sa.BigInteger, primary_key=True, unique=True)
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
        raise NotImplementedError("SQLGraph does not support edges")

    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> np.ndarray:
        raise NotImplementedError("SQLGraph does not support filtering nodes by attribute")

    def node_ids(self) -> np.ndarray:
        raise NotImplementedError("SQLGraph does not support node ids")

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
        raise NotImplementedError("SQLGraph does not support time points")

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError("SQLGraph does not support node features")

    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
        include_targets: bool = False,
    ) -> pl.DataFrame:
        raise NotImplementedError("SQLGraph does not support edge features")

    @property
    def node_features_keys(self) -> list[str]:
        keys = list(Node.__table__.columns.keys())
        return keys

    @property
    def edge_features_keys(self) -> list[str]:
        raise NotImplementedError("SQLGraph does not support edge features keys")

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
        LOG.info("`add_node_feature_key` statement: %s", add_column_stmt)

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

    def update_node_features(
        self,
        *,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        raise NotImplementedError("SQLGraph does not support updating node features")

    def update_edge_features(
        self,
        *,
        edge_ids: ArrayLike,
        attributes: dict[str, Any],
    ) -> None:
        raise NotImplementedError("SQLGraph does not support updating edge features")
