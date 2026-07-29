from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def pages() -> list[Document]:
    """Two fake paper pages with the metadata the pipeline depends on."""
    return [
        Document(
            page_content=(
                "Scaled dot-product attention computes the dot products of the query with all "
                "keys, divides each by the square root of d_k, and applies a softmax function "
                "to obtain the weights on the values. " * 6
            ),
            metadata={
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762",
                "page": 4,
                "total_pages": 15,
                "source": "1706_03762.pdf",
                "source_path": "/tmp/1706_03762.pdf",
            },
        ),
        Document(
            page_content=(
                "LoRA freezes the pretrained model weights and injects trainable rank "
                "decomposition matrices into each layer of the Transformer architecture. " * 6
            ),
            metadata={
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "arxiv_id": "2106.09685",
                "page": 2,
                "total_pages": 26,
                "source": "2106_09685.pdf",
                "source_path": "/tmp/2106_09685.pdf",
            },
        ),
    ]
