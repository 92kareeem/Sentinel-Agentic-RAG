"""Fetch the quantized ONNX MiniLM into models/onnx/ and prove parity.

Downloads the community ONNX export of the exact model we index with
(Xenova/all-MiniLM-L6-v2, int8-quantized, ~23 MB), then embeds test strings
with BOTH backends and asserts cosine similarity — if the ONNX query vectors
didn't live in the same space as our torch-built FAISS index, retrieval would
be confidently wrong.
"""

import numpy as np
from huggingface_hub import hf_hub_download

from app.config import get_settings
from app.rag import embeddings

REPO = "Xenova/all-MiniLM-L6-v2"


def main() -> None:
    settings = get_settings()
    dest = settings.onnx_model_dir
    dest.mkdir(parents=True, exist_ok=True)

    for remote, local in [("onnx/model_quantized.onnx", "model_quantized.onnx"),
                          ("tokenizer.json", "tokenizer.json")]:
        path = hf_hub_download(REPO, remote)
        (dest / local).write_bytes(open(path, "rb").read())
        print(f"fetched {local} ({(dest / local).stat().st_size / 1e6:.1f} MB)")

    tests = [
        "What are the refund conditions for defective items?",
        "Engineer laptop budget refresh cycle",
        "escalation after 10 business days",
    ]
    torch_vecs = embeddings.embed_texts(tests)  # default backend
    onnx_vecs = embeddings._embed_onnx(tests)
    sims = (torch_vecs * onnx_vecs).sum(axis=1)  # both normalized -> cosine
    for text, sim in zip(tests, sims, strict=True):
        print(f"cosine(torch, onnx) = {sim:.4f}  | {text[:50]}")
    assert sims.min() > 0.98, "ONNX embeddings diverge from the index's space!"
    print("PARITY OK — onnx backend is safe against the torch-built index")


if __name__ == "__main__":
    main()
