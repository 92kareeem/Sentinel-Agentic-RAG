from app.agents import retriever
from app.models.schemas import Chunk


def test_rerank_prefers_exact_section_and_metadata_matches() -> None:
    chunks = [
        Chunk(
            chunk_id="d1_s0_c0",
            doc_id="d1",
            section_path="Refund policy",
            text="Refunds take 30 days",
            is_table=False,
            token_count=4,
            char_start=0,
            char_end=20,
        ),
        Chunk(
            chunk_id="d1_s1_c0",
            doc_id="d1",
            section_path="FAQ",
            text="The refund window is 30 days",
            is_table=False,
            token_count=5,
            char_start=21,
            char_end=45,
        ),
        Chunk(
            chunk_id="d2_s0_c0",
            doc_id="d2",
            section_path="General",
            text="This is unrelated content",
            is_table=False,
            token_count=4,
            char_start=0,
            char_end=25,
        ),
    ]

    ranked = retriever._rerank_candidates(
        "refund policy 30 days",
        chunks,
        [0, 1, 2],
        query_terms={"refund", "policy", "30", "days"},
    )

    assert ranked[0].chunk_id == "d1_s0_c0"
    assert ranked[1].chunk_id == "d1_s1_c0"
