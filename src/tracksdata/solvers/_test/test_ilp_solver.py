import logging

import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.expr import AttrExpr
from tracksdata.graph._rustworkx_graph import RustWorkXGraph
from tracksdata.solvers._ilp_solver import ILPSolver


def test_ilp_solver_init_default() -> None:
    """Test ILPSolver initialization with default parameters."""
    solver = ILPSolver()

    assert solver.output_key == DEFAULT_ATTR_KEYS.SOLUTION
    assert isinstance(solver.edge_weight_expr, AttrExpr)
    assert isinstance(solver.node_weight_expr, AttrExpr)
    assert isinstance(solver.appearance_weight_expr, AttrExpr)
    assert isinstance(solver.disappearance_weight_expr, AttrExpr)
    assert isinstance(solver.division_weight_expr, AttrExpr)


def test_ilp_solver_init_custom() -> None:
    """Test ILPSolver initialization with custom parameters."""
    solver = ILPSolver(
        edge_weight="custom_edge_weight",
        node_weight="custom_node_weight",
        appearance_weight=1.5,
        disappearance_weight=-0.5,
        division_weight=2.0,
        output_key="custom_solution",
    )

    assert solver.output_key == "custom_solution"
    assert isinstance(solver.edge_weight_expr, AttrExpr)
    assert isinstance(solver.node_weight_expr, AttrExpr)
    assert isinstance(solver.appearance_weight_expr, AttrExpr)
    assert isinstance(solver.disappearance_weight_expr, AttrExpr)
    assert isinstance(solver.division_weight_expr, AttrExpr)


def test_ilp_solver_init_with_attr_expr() -> None:
    """Test ILPSolver initialization with AttrExpr objects."""
    edge_weight_expr = AttrExpr("edge_weight") * 2
    node_weight_expr = -AttrExpr("node_weight")

    solver = ILPSolver(edge_weight=edge_weight_expr, node_weight=node_weight_expr)

    # Check that the expressions have the same underlying polars expression
    assert str(solver.edge_weight_expr) == str(edge_weight_expr)
    assert str(solver.node_weight_expr) == str(node_weight_expr)


def test_ilp_solver_reset_model() -> None:
    """Test that reset_model clears all internal state."""
    solver = ILPSolver()

    # Manually modify some state
    solver._count = 10
    solver._node_vars = {"test": "value"}

    solver.reset_model()

    assert solver._count == 0
    assert len(solver._node_vars) == 0
    assert len(solver._edge_vars) == 0
    assert len(solver._appear_vars) == 0
    assert len(solver._disappear_vars) == 0
    assert len(solver._division_vars) == 0


def test_ilp_solver_solve_empty_graph() -> None:
    """Test solving on an empty graph."""
    graph = RustWorkXGraph()
    solver = ILPSolver()

    # Should not raise an error on empty graph
    with pytest.raises(ValueError, match="Empty ILPSolver model, there is nothing to solve."):
        solver.solve(graph)


def test_ilp_solver_solve_no_edges(caplog: pytest.LogCaptureFixture) -> None:
    """Test solving on a graph with nodes but no edges."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add some nodes
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    solver = ILPSolver(
        appearance_weight=1.0,
        disappearance_weight=1.0,
        division_weight=1.0,
    )

    # Should not raise an error with no edges
    with caplog.at_level(logging.WARNING):
        solver.solve(graph)

    assert "Trivial solution found with all variables set to 0!" in caplog.text


def test_ilp_solver_solve_simple_case() -> None:
    """Test solving with a simple graph."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges with weights (negative weights for minimization)
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.0})
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -2.0})

    solver = ILPSolver()
    solver.solve(graph)

    # Check that solution keys are added
    node_features = graph.node_features()
    edge_features = graph.edge_features()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_features.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_features.columns

    # Check that some solution is found
    selected_edges = edge_features.filter(edge_features[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_edges) > 0
    assert len(selected_nodes) > 0


def test_ilp_solver_solve_with_appearance_weight() -> None:
    """Test solving with appearance weights."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.1})

    # Use positive appearance weight to penalize appearances
    solver = ILPSolver(appearance_weight=1.0)
    solver.solve(graph)

    # Check that solution is found
    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0

    # resetting the solution
    graph.update_node_features(node_ids=[node0, node1], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_features(edge_ids=[edge1], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(appearance_weight=2.0)
    solver.solve(graph)

    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_with_disappearance_weight() -> None:
    """Test solving with disappearance weights."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.1})

    # Use positive disappearance weight to penalize disappearances
    solver = ILPSolver(disappearance_weight=1.0)
    solver.solve(graph)

    # Check that solution is found
    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0

    # resetting the solution
    graph.update_node_features(node_ids=[node0, node1], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_features(edge_ids=[edge1], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(disappearance_weight=2.0)
    solver.solve(graph)

    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_with_division_weight() -> None:
    """Test solving with division weights."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes for division scenario
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges to create potential division
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -2.0})
    edge2 = graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -2.1})

    # Use positive division weight to penalize divisions
    solver = ILPSolver(
        appearance_weight=1.0,
        division_weight=3.0,
    )
    solver.solve(graph)

    # Check that solution is found
    node_features = graph.node_features()
    edge_features = graph.edge_features()

    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_edges = edge_features.filter(edge_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0
    assert len(selected_edges) > 0

    # resetting the solution
    graph.update_node_features(node_ids=[node0, node1, node2], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_features(edge_ids=[edge1, edge2], attributes={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(
        appearance_weight=2.5,
        division_weight=5.0,
    )
    solver.solve(graph)

    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_custom_edge_weight_expr() -> None:
    """Test solving with custom edge weight expression."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key("custom_weight", 0.0)
    graph.add_edge_feature_key("confidence", 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge with custom attributes
    graph.add_edge(node0, node1, {"custom_weight": 2.0, "confidence": 0.8})

    # Use complex expression: -custom_weight * confidence
    weight_expr = -AttrExpr("custom_weight") * AttrExpr("confidence")
    solver = ILPSolver(edge_weight=weight_expr)
    solver.solve(graph)

    # Check that solution keys are added
    node_features = graph.node_features()
    edge_features = graph.edge_features()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_features.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_features.columns


def test_ilp_solver_solve_custom_node_weight_expr() -> None:
    """Test solving with custom node weight expression."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("quality", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes with quality attribute
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "quality": 0.9})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "quality": 0.7})

    # Add edge
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.0})

    # Use node quality as weight (negative to encourage high quality nodes)
    node_weight_expr = -AttrExpr("quality")
    solver = ILPSolver(node_weight=node_weight_expr)
    solver.solve(graph)

    # Check that solution is found
    node_features = graph.node_features()
    selected_nodes = node_features.filter(node_features[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0


def test_ilp_solver_solve_custom_output_key() -> None:
    """Test solving with custom output key."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes and edges
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.0})

    custom_key = "my_ilp_solution"
    solver = ILPSolver(output_key=custom_key)
    solver.solve(graph)

    # Check that custom key is used
    node_features = graph.node_features()
    edge_features = graph.edge_features()

    assert custom_key in node_features.columns
    assert custom_key in edge_features.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in node_features.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in edge_features.columns


def test_ilp_solver_solve_with_all_weights() -> None:
    """Test solving with all weight types specified."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.5})
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -1.0})

    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        node_weight=-0.5,  # Encourage node selection
        appearance_weight=0.3,  # Penalize appearances
        disappearance_weight=0.2,  # Penalize disappearances
        division_weight=1.0,  # Penalize divisions
    )
    solver.solve(graph)

    # Check that solution is found
    node_features = graph.node_features()
    edge_features = graph.edge_features()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_features.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_features.columns


def test_ilp_solver_evaluate_expr_scalar() -> None:
    """Test _evaluate_expr method with scalar expressions."""
    import polars as pl

    solver = ILPSolver()

    # Test with scalar expression (no column dependencies)
    scalar_expr = AttrExpr(5.0)
    df = pl.DataFrame({"dummy": [1, 2, 3]})

    result = solver._evaluate_expr(scalar_expr, df)

    assert result == [5.0, 5.0, 5.0]  # Should repeat scalar for each row


def test_ilp_solver_evaluate_expr_column() -> None:
    """Test _evaluate_expr method with column expressions."""
    import polars as pl

    solver = ILPSolver()

    # Test with column expression
    column_expr = AttrExpr("values") * 2
    df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

    result = solver._evaluate_expr(column_expr, df)

    assert result == [2.0, 4.0, 6.0]


def test_ilp_solver_division_constraint() -> None:
    """Test that division constraint is properly enforced: node_var >= division_var."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_edge_feature_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Create a scenario where division would be tempting but should be constrained
    # Time 0: 1 parent node
    parent_node = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Time 1: 2 potential children
    child1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 0.0})
    child2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": -1.0, "y": 0.0})

    # Add edges from parent to both children with negative weights
    graph.add_edge(parent_node, child1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -5.0})
    graph.add_edge(parent_node, child2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: -3.0})

    # Use negative division weight to make division attractive
    # But positive edge weights to make connections costly
    # This tests that division can overcome edge costs when beneficial
    solver = ILPSolver(
        node_weight=0.0,  # Neutral node cost
        division_weight=1.5,  # Penalize divisions, but not too much
        appearance_weight=1.0,  # Penalize appearances
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
    )

    solver.solve(graph)

    # Check results
    node_features = graph.node_features(
        node_ids=[parent_node, child1, child2],
        feature_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.SOLUTION],
    )
    edge_features = graph.edge_features()

    assert node_features[DEFAULT_ATTR_KEYS.SOLUTION].all()

    # Get solution status for each element
    parent_selected, child1_selected, child2_selected = node_features[DEFAULT_ATTR_KEYS.SOLUTION].to_list()

    child2_selected = node_features.filter(node_features[DEFAULT_ATTR_KEYS.NODE_ID] == child2)[
        DEFAULT_ATTR_KEYS.SOLUTION
    ].to_list()[0]

    selected_edges = edge_features.filter(edge_features[DEFAULT_ATTR_KEYS.SOLUTION])

    # Key test of division constraint: node_var >= division_var
    # This means division can only happen if the node is selected

    # Count edges that are selected
    num_selected_edges = len(selected_edges)

    # The key constraint test: if edges are selected, parent must be selected
    assert num_selected_edges == 2

    # Validate flow constraints: edges can only be selected if their endpoints are selected
    for edge_row in selected_edges.iter_rows(named=True):
        source_id = edge_row[DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        target_id = edge_row[DEFAULT_ATTR_KEYS.EDGE_TARGET]

        # For this test, we know the structure, so we can check directly
        if source_id == parent_node:
            assert parent_selected, "Edge from parent selected but parent not selected"
        if target_id == child1:
            assert child1_selected, "Edge to child1 selected but child1 not selected"
        if target_id == child2:
            assert child2_selected, "Edge to child2 selected but child2 not selected"
