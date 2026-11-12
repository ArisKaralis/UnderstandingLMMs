from fastapi.testclient import TestClient

from backend.app.main import app

c = TestClient(app)


def test_specs_and_graph():
    s = c.get("/api/specs").json()
    keys = [m["key"] for m in s["models"]]
    assert "qwen3.8-27b" in keys and not any("235b" in k or "122b" in k for k in keys)
    g = c.get("/api/graph/qwen3.5-0.8b").json()
    assert g["context"]["n_vision"] == 196 and g["summary"]["params_total"] > 8e8


def test_playground_overrides_change_params():
    base = c.get("/api/graph/qwen3-vl-2b").json()["summary"]["params_total"]
    half = c.get("/api/graph/qwen3-vl-2b", params={"overrides": '{"text.num_layers": 14}'}).json()
    assert half["summary"]["params_total"] < base
    assert sum(1 for n in half["nodes"] if n["level"] == "block" and n["id"].startswith("llm.")) == 14


def test_traces_api():
    t = c.get("/api/traces").json()
    if not t:
        return
    tid = t[0]["id"]
    for part in ("manifest", "tokens", "graph", "outputs", "stats", "previews"):
        assert c.get(f"/api/traces/{tid}/{part}").status_code == 200
    r = c.get(f"/api/traces/{tid}/preview/residual_norm").json()
    assert len(r["shape"]) == 2
    # traversal attempts that reach the handler must not escape the bundle
    assert c.get(f"/api/traces/{tid}/preview/..%2F..%2Fmanifest").status_code in (404, 400)
    r = c.get("/..%2F..%2Fpyproject.toml")  # SPA fallback must not serve files outside frontend/dist
    assert "lmmviz" not in r.text
    assert c.get("/api/training/..%2Ftraces/run.json").status_code in (404, 400)
