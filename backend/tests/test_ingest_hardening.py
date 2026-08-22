import time

from app.rag import ingest_runtime


def test_submit_ingest_job_tracks_completed_status(tmp_path, monkeypatch) -> None:
    local_path = tmp_path / "doc.txt"
    local_path.write_text("hello world", encoding="utf-8")

    def fake_merge(path):
        assert path == local_path
        return 2, "v2"

    monkeypatch.setattr(ingest_runtime, "merge_document", fake_merge)

    job_id = ingest_runtime.submit_ingest_job(local_path, "doc-123", "doc.txt")
    status = ingest_runtime.get_job_status(job_id)

    assert status["doc_id"] == "doc-123"
    assert status["filename"] == "doc.txt"

    for _ in range(50):
        status = ingest_runtime.get_job_status(job_id)
        if status["status"] == "completed":
            break
        time.sleep(0.05)

    status = ingest_runtime.get_job_status(job_id)
    assert status["status"] == "completed"
    assert status["chunks_indexed"] == 2
    assert status["index_version"] == "v2"
