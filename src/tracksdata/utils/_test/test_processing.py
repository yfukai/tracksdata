from collections.abc import Generator

from tqdm import tqdm

from tracksdata.utils._processing import maybe_show_progress


def test_maybe_show_progress_with_progress() -> None:
    """Test maybe_show_progress when show_progress=True."""
    data = [1, 2, 3, 4, 5]

    result = maybe_show_progress(data, show_progress=True)

    # Should return a tqdm object
    assert isinstance(result, tqdm)

    # Should be able to iterate through it
    collected = list(result)
    assert collected == data


def test_maybe_show_progress_without_progress() -> None:
    """Test maybe_show_progress when show_progress=False."""
    data = [1, 2, 3, 4, 5]

    result = maybe_show_progress(data, show_progress=False)

    # Should return the original iterable
    assert result is data

    # Should be able to iterate through it
    collected = list(result)
    assert collected == data


def test_maybe_show_progress_with_tqdm_kwargs() -> None:
    """Test maybe_show_progress with additional tqdm kwargs."""
    data = range(10)

    result = maybe_show_progress(data, show_progress=True, desc="Test progress", total=10, unit="items")

    # Should return a tqdm object
    assert isinstance(result, tqdm)

    # Check that the description was set
    assert result.desc == "Test progress"
    assert result.unit == "items"
    assert result.total == 10

    # Should be able to iterate through it
    collected = list(result)
    assert collected == list(data)


def test_maybe_show_progress_with_generator() -> None:
    """Test maybe_show_progress with a generator."""

    def data_generator() -> Generator[int, None, None]:
        for i in range(5):
            yield i * 2

    # With progress
    result_with_progress = maybe_show_progress(data_generator(), show_progress=True)
    assert isinstance(result_with_progress, tqdm)
    collected_with_progress = list(result_with_progress)
    assert collected_with_progress == [0, 2, 4, 6, 8]

    # Without progress
    result_without_progress = maybe_show_progress(data_generator(), show_progress=False)
    collected_without_progress = list(result_without_progress)
    assert collected_without_progress == [0, 2, 4, 6, 8]


def test_maybe_show_progress_empty_iterable() -> None:
    """Test maybe_show_progress with empty iterable."""
    data = []

    # With progress
    result_with_progress = maybe_show_progress(data, show_progress=True)
    assert isinstance(result_with_progress, tqdm)
    assert list(result_with_progress) == []

    # Without progress
    result_without_progress = maybe_show_progress(data, show_progress=False)
    assert result_without_progress is data
    assert list(result_without_progress) == []


def test_maybe_show_progress_string_iterable() -> None:
    """Test maybe_show_progress with string (which is iterable)."""
    data = "hello"

    # With progress
    result_with_progress = maybe_show_progress(data, show_progress=True)
    assert isinstance(result_with_progress, tqdm)
    collected_with_progress = list(result_with_progress)
    assert collected_with_progress == ["h", "e", "l", "l", "o"]

    # Without progress
    result_without_progress = maybe_show_progress(data, show_progress=False)
    assert result_without_progress is data
    collected_without_progress = list(result_without_progress)
    assert collected_without_progress == ["h", "e", "l", "l", "o"]
