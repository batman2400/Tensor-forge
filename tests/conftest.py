from __future__ import annotations

import pytest

from tests.helpers import client_for


@pytest.fixture
def client():
    with client_for() as api:
        yield api
