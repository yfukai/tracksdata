# Examples

This section provides practical examples of using TracksData for multi-object tracking tasks.

## Basic Tracking Example

Here's a complete basic example that demonstrates the core workflow of TracksData. This example is available as an executable Python file at [`docs/examples/basic.py`](examples/basic.py).

```python
--8<-- "docs/examples/basic.py"
```

### Alternative: ILP Solver

For more sophisticated tracking, you can replace the NearestNeighborsSolver with the ILPSolver:

```python
# Alternative solver with more parameters
from tracksdata.solvers import ILPSolver

dist_weight = 1 / dist_operator.distance_threshold
solver = ILPSolver(
    edge_weight=-td.EdgeAttr("iou") + td.EdgeAttr("weight") * dist_weight,
    node_weight=0.0,
    appearance_weight=10.0,
    disappearance_weight=10.0,
    division_weight=1.0,
)
solver.solve(graph)
```

### Visualization with Napari

To visualize the tracking results:

```python
import napari

# After running the tracking example above
viewer = napari.Viewer()
viewer.add_labels(labels)
viewer.add_tracks(tracks_df, graph=track_graph)
napari.run()
```

## Key Components Explained

- **Graph**: The core data structure holding nodes (objects) and edges (connections)
- **Nodes Operators**: Extract object features from segmented images (RegionPropsNodes, MaskNodes, etc.)
- **Edges Operators**: Create temporal connections between objects (DistanceEdges, IoUEdges, etc.)
- **Solvers**: Optimize tracking assignments (NearestNeighborsSolver, ILPSolver)
- **Functional**: Utilities for format conversion and visualization

## Next Steps

- Check the [Getting Started](getting_started.md) guide for more detailed explanations
- Explore the [Concepts](concepts.md) page to understand the architecture
- See the API reference for complete documentation of all components
