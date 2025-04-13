import abc

# NOTE:
# - maybe a single basegraph is better
# - nodes have a t, and space

class BaseReadOnlyGraph(abc.ABC):
    """
    Base class for viewing a graph.
    """


class BaseWritableGraph(BaseReadOnlyGraph):
    """
    Base class for writing to a graph.
    """

    # TODO


class BaseGraphBackend(abc.ABC):
    """
    Base class for a graph backend.
    """
    # TODO
