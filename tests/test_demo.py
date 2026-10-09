"""The demo is public HTML. The API key is not in the page."""

from __future__ import annotations

from tests.helpers import client_for


def test_demo_pages_are_public_and_contain_no_key():
    with client_for() as client:
        page = client.get("/demo/")
        assert page.status_code == 200, page.text
        assert page.headers["content-type"].startswith("text/html")
        assert "session storage" in page.text
        assert "tf2_" not in page.text
        assert '<script src="app.js">' in page.text
        script = client.get("/demo/app.js")
        assert script.status_code == 200
        assert "tf2_" not in script.text
        assert "sessionStorage" in script.text
        metrics = client.get("/demo/metrics.html")
        assert metrics.status_code == 200
        assert "tf2_" not in metrics.text
        diagram = client.get("/demo/calibration.svg")
        assert diagram.status_code == 200
        scores = client.get("/demo/metrics.json")
        assert scores.status_code == 200
        assert scores.json()["macro_f1"] > 0.75


def test_demo_post_is_405_without_asking_for_a_key(client):
    response = client.post("/demo/", json={"channel": "chat", "text": "hi"})
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
