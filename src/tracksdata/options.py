"""Global options system for TracksData."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# Module-private mutable state
_options_stack: list["Options"] = []


@dataclass
class Options:
    """
    Global options for TracksData.

    This class provides a centralized way to control various behaviors
    across the library, such as progress display and multiprocessing.

    Parameters
    ----------
    show_progress : bool, default True
        Whether to display progress bars during operations.
    """

    show_progress: bool = True

    def __enter__(self) -> "Options":
        """Enter the context manager."""
        # Push current options to stack
        _options_stack.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager."""
        # Pop this options from stack
        if _options_stack:
            _options_stack.pop()


# Default options - always at the bottom of the stack
_default_options = Options()


def get_options() -> Options:
    """
    Get the current global options.

    Returns
    -------
    Options
        The current global options instance.
    """
    # Return the top of the stack, or default if stack is empty
    return _options_stack[-1] if _options_stack else _default_options


def set_options(options: Options) -> None:
    """
    Set the global options.

    Parameters
    ----------
    options : Options
        The options to set as global.
    """
    global _default_options
    _default_options = options


@contextmanager
def options_context(**kwargs: Any) -> Generator[Options, None, None]:
    """
    Context manager for temporarily modifying options.

    Parameters
    ----------
    **kwargs : Any
        Options parameters to temporarily set.

    Yields
    ------
    Options
        The options object with the temporary settings.

    Examples
    --------
    ```python
    from tracksdata.options import options_context

    with options_context(show_progress=False):
        # Operations here will not show progress
        pass
    ```

    See Also
    --------
    [Options][tracksdata.options.Options]:
        The global options class.

    """
    temp_options = Options(**kwargs)
    with temp_options:
        yield temp_options
