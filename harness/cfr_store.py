"""
harness/cfr_store.py — Local CFR vector store for the RetrievalAgent.

Index is built once by scripts/build_cfr_index.py and stored in
data/cfr_index/ (gitignored). At query time, the store embeds the query
with the same sentence-transformers model and returns top-k sections by
cosine similarity (pure numpy, no FAISS dependency).

If the index directory does not exist, an informative error is raised
pointing to the build script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "data" / "cfr_index"
_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class CFRSection:
    part: str          # e.g. "11"
    section_id: str    # e.g. "11.10"
    title: str         # e.g. "Controls for closed systems."
    text: str          # full section text


class CFRStore:
    """
    Thin wrapper around a pre-built numpy embedding index.

    Parameters
    ----------
    index_dir : path to the directory containing embeddings.npy and sections.jsonl.
                Defaults to data/cfr_index/ relative to the repo root.
    """

    def __init__(self, index_dir: str | Path = _DEFAULT_INDEX_DIR) -> None:
        self._index_dir = Path(index_dir)
        self._embeddings: np.ndarray | None = None
        self._sections: list[CFRSection] | None = None
        self._model: "SentenceTransformer | None" = None

    def _load_index(self) -> None:
        if self._sections is not None:
            return
        emb_path = self._index_dir / "embeddings.npy"
        sec_path = self._index_dir / "sections.jsonl"
        if not emb_path.exists() or not sec_path.exists():
            raise FileNotFoundError(
                f"CFR index not found at {self._index_dir}.\n"
                "Build it first:  python3 scripts/build_cfr_index.py"
            )
        self._embeddings = np.load(str(emb_path))
        sections: list[CFRSection] = []
        with sec_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    sections.append(CFRSection(**obj))
        self._sections = sections

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def retrieve(self, query: str, k: int = 3) -> list[CFRSection]:
        """Return the top-k CFR sections most relevant to `query`."""
        self._load_index()
        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        assert self._embeddings is not None
        sims = self._embeddings @ q_vec  # (n_sections,) — dot of unit vecs = cosine
        top_k_idx = np.argsort(sims)[::-1][:k]
        assert self._sections is not None
        return [self._sections[i] for i in top_k_idx]

    @property
    def n_sections(self) -> int:
        self._load_index()
        return len(self._sections or [])
