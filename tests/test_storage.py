from edge_ai.storage import HistoryStore


def test_history_round_trip(tmp_path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.add("healthy", 0.8, "demo")
    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["predicted_class"] == "healthy"
    store.clear()
    assert store.list() == []

