"""Internal filter AST shared by graph backends.

Each `Attr` carries an optional `_FilterNode` that records the structured form
of a boolean filter expression (leaf comparison or compound). Backends walk
this AST to translate filters into SQL clauses, polars predicates, or Python
dict checks.

The AST is intentionally minimal: leaf comparisons hold `(column, op, other)`
plus the originating `Attr` subclass so node/edge dispatch survives, and
compound nodes hold a logical op and a list of children.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tracksdata.attrs import Attr


CompoundOp = Literal["and", "or", "xor", "not"]


@dataclass(frozen=True)
class _FilterLeaf:
    """A single column comparison: ``column op other``.

    `kind` is the originating `Attr` subclass (`NodeAttr` / `EdgeAttr` / `Attr`)
    used by backend dispatch to decide which graph table the filter targets.
    """

    column: str
    op: Callable
    other: object
    kind: type[Attr]


@dataclass(frozen=True)
class _FilterCompound:
    """A boolean combination of filter nodes."""

    op: CompoundOp
    operands: tuple[_FilterNode, ...]


_FilterNode = _FilterLeaf | _FilterCompound


def walk_leaves(node: _FilterNode) -> Iterable[_FilterLeaf]:
    """Yield all leaf comparisons under `node` in left-to-right order."""
    if isinstance(node, _FilterLeaf):
        yield node
        return
    for child in node.operands:
        yield from walk_leaves(child)
