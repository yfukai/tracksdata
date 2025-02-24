
import abc


class BaseReadOnlyGraph(abc.ABC):
    """
    Base class for viewing a graph.
    """
    # TODO


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
    