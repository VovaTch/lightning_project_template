from typing import Any

import pytest

from utils.others import load_config


@pytest.fixture
def cfg() -> dict[str, Any]:
    cfg_path = "tests/test_config.yaml"
    return load_config(cfg_path)
