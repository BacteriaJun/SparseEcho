import json

from sparseecho.runtime import JsonlTelemetrySink


def test_jsonl_telemetry_is_append_only(tmp_path):
    path = tmp_path / "runtime.jsonl"
    sink = JsonlTelemetrySink(path)
    sink.emit({"event": "one", "plan_fingerprint": "0" * 64})
    sink.emit({"event": "two", "plan_fingerprint": "0" * 64})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["event"] for line in lines] == ["one", "two"]
