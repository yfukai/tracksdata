"""Benchmark: cold-frame fetch for GraphArrayView with NDChunkCache.

Standalone script (not part of ASV). Measures the wall time, number of SQL
executes, and number of blosc2 tensor unpacks per scenario, to validate the
NDChunkCache chunk-coalescing fix that collapses N per-chunk SQL queries into
a single bounding-box query per cold frame fetch.

Usage::

    PYTHONPATH=<worktree>/src \
    /workspace/.venv/bin/python benchmarks/perf_chunk_coalescing.py
"""

from __future__ import annotations

import os
import sys
import time
import types

import numpy as np


# ---------------------------------------------------------------------------
# Stub out spatial_graph.PointRTree and ilpy so this script can run in an
# environment without compiled Cython extensions or the ILP solver.
# ---------------------------------------------------------------------------
class _PointRTree:
    def __init__(self, item_dtype: str = "int64", coord_dtype: str = "float32", dims: int = 2):
        self._item_dtype = np.dtype(item_dtype)
        self._coord_dtype = np.dtype(coord_dtype)
        self._dims = int(dims)
        self._items: list[int] = []
        self._mins: list[np.ndarray] = []
        self._maxs: list[np.ndarray] = []

    @property
    def dims(self) -> int:
        return self._dims

    def insert_point_items(self, items, positions) -> None:
        positions = np.asarray(positions, dtype=self._coord_dtype)
        for it, p in zip(np.asarray(items).tolist(), positions, strict=True):
            self._items.append(int(it))
            self._mins.append(np.asarray(p, dtype=self._coord_dtype).copy())
            self._maxs.append(np.asarray(p, dtype=self._coord_dtype).copy())

    def insert_bb_items(self, items, bb_mins, bb_maxs) -> None:
        bb_mins = np.asarray(bb_mins, dtype=self._coord_dtype)
        bb_maxs = np.asarray(bb_maxs, dtype=self._coord_dtype)
        for it, lo, hi in zip(np.asarray(items).tolist(), bb_mins, bb_maxs, strict=True):
            self._items.append(int(it))
            self._mins.append(np.asarray(lo, dtype=self._coord_dtype).copy())
            self._maxs.append(np.asarray(hi, dtype=self._coord_dtype).copy())

    def delete_items(self, items, bb_mins, bb_maxs=None) -> None:
        bb_mins = np.asarray(bb_mins, dtype=self._coord_dtype)
        bb_maxs = bb_mins if bb_maxs is None else np.asarray(bb_maxs, dtype=self._coord_dtype)
        for it, lo, hi in zip(np.asarray(items).tolist(), bb_mins, bb_maxs, strict=True):
            it = int(it)
            for i, (cit, cl, ch) in enumerate(zip(self._items, self._mins, self._maxs, strict=True)):
                if cit == it and np.array_equal(cl, lo) and np.array_equal(ch, hi):
                    del self._items[i]
                    del self._mins[i]
                    del self._maxs[i]
                    break

    def search(self, bb_min, bb_max) -> np.ndarray:
        lo = np.asarray(bb_min, dtype=self._coord_dtype).reshape(-1)
        hi = np.asarray(bb_max, dtype=self._coord_dtype).reshape(-1)
        out = []
        for it, cl, ch in zip(self._items, self._mins, self._maxs, strict=True):
            if np.all(ch >= lo) and np.all(cl <= hi):
                out.append(it)
        return np.asarray(out, dtype=self._item_dtype)


def _install_stubs() -> None:
    if "spatial_graph" not in sys.modules:
        mod = types.ModuleType("spatial_graph")
        mod.PointRTree = _PointRTree
        mod.SpatialGraph = type("SpatialGraph", (), {})
        mod.SpatialDiGraph = type("SpatialDiGraph", (), {})
        sys.modules["spatial_graph"] = mod
    else:
        sys.modules["spatial_graph"].PointRTree = _PointRTree

    try:
        import ilpy
    except ImportError:
        ilpy = types.ModuleType("ilpy")
        for name in (
            "Constraint",
            "Constraints",
            "Objective",
            "Preference",
            "Relation",
            "Sense",
            "Solver",
            "Solution",
            "SolverStatus",
            "Variable",
            "VariableType",
            "Binary",
            "Integer",
            "Continuous",
            "Maximize",
            "Minimize",
            "Equal",
            "LessEqual",
            "GreaterEqual",
            "Any",
            "EventData",
            "Expression",
            "Gurobi",
            "Scip",
            "SolverBackend",
        ):
            setattr(ilpy, name, type(name, (), {}))
        sys.modules["ilpy"] = ilpy


_install_stubs()


# ---------------------------------------------------------------------------
# Imports that rely on tracksdata + sqlalchemy + blosc2.
# ---------------------------------------------------------------------------
import blosc2  # noqa: E402
import polars as pl  # noqa: E402
import sqlalchemy as sa  # noqa: E402

from tracksdata.array import GraphArrayView  # noqa: E402
from tracksdata.constants import DEFAULT_ATTR_KEYS  # noqa: E402
from tracksdata.graph import SQLGraph  # noqa: E402
from tracksdata.nodes._mask import Mask  # noqa: E402

# ---------------------------------------------------------------------------
# Counter hooks.
# ---------------------------------------------------------------------------
_EXEC_COUNT = 0
_UNPACK_COUNT = 0
_ORIG_UNPACK = blosc2.unpack_tensor


def _counting_unpack(*args, **kwargs):
    global _UNPACK_COUNT
    _UNPACK_COUNT += 1
    return _ORIG_UNPACK(*args, **kwargs)


blosc2.unpack_tensor = _counting_unpack


def _hook_engine(engine: sa.Engine) -> None:
    @sa.event.listens_for(engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        global _EXEC_COUNT
        _EXEC_COUNT += 1


def _reset_counters() -> None:
    global _EXEC_COUNT, _UNPACK_COUNT
    _EXEC_COUNT = 0
    _UNPACK_COUNT = 0


# ---------------------------------------------------------------------------
# Benchmark driver.
# ---------------------------------------------------------------------------
DB_PATH = "/tmp/perf_chunk_coalescing.db"
FRAMES = 50
NODES_PER_FRAME = 50
MASK_SIZE = 32
FRAME_SIZE = 512
CHUNK = 128
SHAPE = (FRAMES, FRAME_SIZE, FRAME_SIZE)


def _build_graph() -> SQLGraph:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    graph = SQLGraph(drivername="sqlite", database=DB_PATH, overwrite=True)
    graph.add_node_attr_key("label", dtype=pl.Int64)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))

    rng = np.random.default_rng(0)
    label = 1
    for t in range(FRAMES):
        # Spread nodes uniformly across the frame.
        positions = rng.integers(0, FRAME_SIZE - MASK_SIZE, size=(NODES_PER_FRAME, 2))
        for y, x in positions:
            mask_data = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
            mask = Mask(mask_data, bbox=np.array([y, x, y + MASK_SIZE, x + MASK_SIZE]))
            graph.add_node(
                {
                    DEFAULT_ATTR_KEYS.T: int(t),
                    "label": label,
                    DEFAULT_ATTR_KEYS.MASK: mask,
                    DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
                },
                validate_keys=False,
            )
            label += 1
    return graph


def _run_scenario(name: str, fn, *, executes_expected: int | None = None) -> tuple[float, int, int]:
    _reset_counters()
    start = time.perf_counter()
    fn()
    wall_ms = (time.perf_counter() - start) * 1e3
    print(f"  {name:<32s}  wall_ms={wall_ms:9.2f}  execute_count={_EXEC_COUNT:4d}  unpack_count={_UNPACK_COUNT:5d}")
    if executes_expected is not None and _EXEC_COUNT != executes_expected:
        raise AssertionError(f"scenario '{name}': expected {executes_expected} executes, got {_EXEC_COUNT}")
    return wall_ms, _EXEC_COUNT, _UNPACK_COUNT


def main() -> None:
    print(f"Building graph in {DB_PATH} ({FRAMES} frames x {NODES_PER_FRAME} nodes x {MASK_SIZE}^2 masks)...")
    t0 = time.perf_counter()
    graph = _build_graph()
    print(f"  built in {time.perf_counter() - t0:.2f} s")

    _hook_engine(graph._engine)
    view = GraphArrayView(
        graph=graph,
        shape=SHAPE,
        attr_key="label",
        chunk_shape=(CHUNK, CHUNK),
    )

    # Capture once at the start: number of chunks per frame = (FRAME/CHUNK)^2.
    chunks_per_frame = (FRAME_SIZE // CHUNK) ** 2
    print(f"chunks per frame = {chunks_per_frame}  (expected 1 execute per cold frame after fix)\n")

    print("=== scenarios ===")
    _run_scenario(
        "cold whole frame t=10",
        lambda: np.asarray(view[10]),
        executes_expected=1,
    )
    _run_scenario(
        "cached whole frame t=10",
        lambda: np.asarray(view[10]),
        executes_expected=0,
    )
    _run_scenario(
        "cold one chunk t=11",
        lambda: np.asarray(view[11, :CHUNK, :CHUNK]),
        executes_expected=1,
    )
    _run_scenario(
        "rest of t=11",
        lambda: np.asarray(view[11]),
        executes_expected=1,
    )

    def cold_three_frames() -> None:
        for t in (12, 13, 14):
            np.asarray(view[t])

    _run_scenario(
        "3 cold frames t=12..14",
        cold_three_frames,
        executes_expected=3,
    )

    print("\nOK: coalescing produces 1 SQL execute per cold frame request.")


if __name__ == "__main__":
    main()
