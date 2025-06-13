import os
import zipfile
from pathlib import Path

import pytest
import requests

from tracksdata.graph import RustWorkXGraph, SQLGraph
from tracksdata.graph._base_graph import BaseGraph


@pytest.fixture(params=[RustWorkXGraph, SQLGraph])
def graph_backend(request) -> BaseGraph:
    """Fixture that provides all implementations of BaseGraph."""
    graph_class: BaseGraph = request.param

    if graph_class == SQLGraph:
        return graph_class(
            drivername="sqlite",
            database=":memory:",
            engine_kwargs={"connect_args": {"check_same_thread": False}},
        )
    else:
        return graph_class()


def _download_to(url: str, output_path: Path) -> None:
    """Download CTC data."""
    with open(output_path, "wb") as f:
        response = requests.get(url)
        response.raise_for_status()
        f.write(response.content)


@pytest.fixture(scope="session")
def ctc_data_dir(pytestconfig: pytest.Config) -> Path:
    """Fixture to download CTC data."""

    temp_dir = Path(pytestconfig.cache._cachedir) / "donwloads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    zip_dir = temp_dir / "Fluo-C2DL-Huh7.zip"
    out_dir = temp_dir / "Fluo-C2DL-Huh7/02_GT/TRA"
    url = "https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip"

    if not zip_dir.exists():
        _download_to(url, zip_dir)

    if not out_dir.exists():
        try:
            with zipfile.ZipFile(zip_dir) as zip_file:
                zip_file.extractall(temp_dir)

        except zipfile.BadZipFile:
            os.remove(zip_dir)
            _download_to(url, zip_dir)
            # retry
            with zipfile.ZipFile(zip_dir) as zip_file:
                zip_file.extractall(temp_dir)

    return out_dir
