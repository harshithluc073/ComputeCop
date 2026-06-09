from __future__ import annotations

from pathlib import Path

from computecop.events import JsonlEventStore


async def test_append_persists_and_tails(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl", max_events=10)
    for index in range(3):
        await store.append("admission.decision", correlation_id=f"c{index}")

    events = await store.read_events()
    assert len(events) == 3
    assert [event["payload"]["correlation_id"] for event in events] == ["c0", "c1", "c2"]
    assert store.persistence_disabled is False
    assert store.disabled_reason is None


async def test_append_flushes_each_line_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    await store.append("policy.yield", reason="ram")

    # The line is durably present immediately after append returns.
    contents = path.read_text(encoding="utf-8")
    assert contents.endswith("\n")
    assert '"policy.yield"' in contents


async def test_tail_limits_returned_events(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    for index in range(5):
        await store.append("admission.decision", correlation_id=f"c{index}")

    tail = await store.tail(limit=2)
    assert [event["payload"]["correlation_id"] for event in tail] == ["c3", "c4"]


async def test_retention_bound_trims_old_events(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl", max_events=2)
    for index in range(4):
        await store.append("admission.decision", correlation_id=f"c{index}")

    events = await store.read_events()
    assert [event["payload"]["correlation_id"] for event in events] == ["c2", "c3"]


async def test_append_disables_persistence_on_oserror(tmp_path: Path) -> None:
    # A file occupies the directory slot, so mkdir of the parent fails with OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    store = JsonlEventStore(blocker / "events.jsonl")

    callbacks: list[tuple[bool, str | None]] = []

    async def on_change(enabled: bool, reason: str | None) -> None:
        callbacks.append((enabled, reason))

    store.set_persistence_callback(on_change)

    # Must not raise even though the path is unwritable.
    await store.append("admission.decision", correlation_id="c0")

    assert store.persistence_disabled is True
    assert store.disabled_reason
    assert callbacks and callbacks[0][0] is False

    # Subsequent appends are silent no-ops and do not re-notify.
    await store.append("admission.decision", correlation_id="c1")
    assert len(callbacks) == 1


async def test_read_events_missing_file_returns_empty(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "missing.jsonl")
    assert await store.read_events() == ()
    assert await store.tail() == ()


async def test_read_events_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    await store.append("policy.yield", reason="ram")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
        handle.write("\n")

    events = await store.read_events()
    assert len(events) == 1
    assert events[0]["kind"] == "policy.yield"
