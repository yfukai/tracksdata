# Examples

This section provides practical examples of using TracksData for multi-object tracking tasks.

## Basic Tracking Example

Here's a complete basic example that demonstrates the core workflow of TracksData. This example is available as an executable Python file at [`docs/examples/basic.py`](examples/basic.py).

```python
--8<-- "docs/examples/basic.py"
```

## Lineage Tree Plotting and Region-Properties Re-computation

A self-contained notebook demonstrating `plot_lineage_tree` (matplotlib lineage trees with
attribute-bound colors and sizes, time windows, and exact timestamps) and `RegionPropsAttrs`
(re-computing region properties from existing masks and an intensity image) is available at
[`docs/examples/lineage_tree_and_regionprops.ipynb`](examples/lineage_tree_and_regionprops.ipynb).

## Key Components Explained

- **Graph**: The core data structure holding nodes (objects) and edges (connections)
- **Nodes Operators**: Extract object features from segmented images (RegionPropsNodes, MaskNodes, etc.)
- **Edges Operators**: Create temporal connections between objects (DistanceEdges, IoUEdges, etc.)
- **Solvers**: Optimize a minimization problem to find the best tracking assignments (NearestNeighborsSolver, ILPSolver)
- **Functional**: Utilities for format conversion and visualization

## Next Steps

- Check the [Getting Started](getting_started.md) guide for more detailed explanations
- Explore the [Concepts](concepts.md) page to understand the architecture
- See the API reference for complete documentation of all components
