import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class IndexedSample:
    doc_id: int
    dataset: str
    split: str
    text: str
    modalities: Set[str]


class ByteWordTokenizer:
    """Simple tokenizer that treats words as tokens and emits byte IDs in [0, 255]."""

    def tokenize_words(self, text: str) -> List[str]:
        return [w.lower() for w in WORD_PATTERN.findall(text)]

    def words_to_bytes(self, words: Iterable[str]) -> List[int]:
        out: List[int] = []
        for word in words:
            out.extend(word.encode("utf-8", errors="ignore"))
            out.append(32)  # keep a delimiter between words
        return [b for b in out if 0 <= b <= 255]


class HFStreamingByteIndexer:
    """
    Build a search-style index over Hugging Face datasets in streaming mode.

    This does not download a local full copy of the dataset; instead, it iterates
    examples remotely and creates:
      1) Word-level inverted index.
      2) Modality map (text/image/audio/video) per sample.
      3) Lightweight co-occurrence graph (auto-KG) from words in the same sample.
      4) Byte histogram + pseudo-weight vector (length 256).
    """

    def __init__(self) -> None:
        self.tokenizer = ByteWordTokenizer()

    @staticmethod
    def _detect_modalities(sample: Dict[str, Any]) -> Set[str]:
        modalities = {"text"}
        if sample.get("image") is not None or sample.get("images") is not None:
            modalities.add("image")
        if sample.get("audio") is not None or sample.get("audios") is not None:
            modalities.add("audio")
        if sample.get("video") is not None or sample.get("videos") is not None:
            modalities.add("video")
        return modalities

    @staticmethod
    def _extract_text(sample: Dict[str, Any], text_fields: Optional[List[str]] = None) -> str:
        if text_fields:
            return " ".join(str(sample.get(field, "")) for field in text_fields).strip()

        for key in ("text", "caption", "question", "answer", "content", "messages"):
            if key not in sample or sample[key] is None:
                continue
            value = sample[key]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                if key == "messages":
                    chunks = []
                    for msg in value:
                        if isinstance(msg, dict):
                            chunks.append(str(msg.get("content", "")))
                        else:
                            chunks.append(str(msg))
                    return " ".join(chunks).strip()
                return " ".join(map(str, value)).strip()
            return str(value)
        return ""

    def build_index(
        self,
        dataset_name: str,
        split: str = "train",
        config_name: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        from datasets import load_dataset

        stream = load_dataset(dataset_name, name=config_name, split=split, streaming=True)

        inverted_index: Dict[str, Set[int]] = defaultdict(set)
        kg_edges: Counter[Tuple[str, str]] = Counter()
        byte_hist = [0] * 256
        samples: List[IndexedSample] = []

        for doc_id, sample in enumerate(islice(stream, max_samples)):
            text = self._extract_text(sample, text_fields=text_fields)
            if not text:
                continue

            words = self.tokenizer.tokenize_words(text)
            if not words:
                continue

            modalities = self._detect_modalities(sample)
            samples.append(
                IndexedSample(
                    doc_id=doc_id,
                    dataset=dataset_name,
                    split=split,
                    text=text,
                    modalities=modalities,
                )
            )

            unique_words = sorted(set(words))
            for token in unique_words:
                inverted_index[token].add(doc_id)

            # auto-KG via co-occurrence links per sample
            for i in range(len(unique_words)):
                for j in range(i + 1, len(unique_words)):
                    kg_edges[(unique_words[i], unique_words[j])] += 1

            for b in self.tokenizer.words_to_bytes(words):
                byte_hist[b] += 1

        total_bytes = max(1, sum(byte_hist))
        pseudo_weights = [count / total_bytes for count in byte_hist]

        return {
            "dataset": dataset_name,
            "split": split,
            "num_indexed_samples": len(samples),
            "samples": [
                {
                    "doc_id": s.doc_id,
                    "dataset": s.dataset,
                    "split": s.split,
                    "modalities": sorted(s.modalities),
                    "text_preview": s.text[:160],
                }
                for s in samples
            ],
            "inverted_index": {token: sorted(ids) for token, ids in inverted_index.items()},
            "kg_edges": [
                {"source": a, "target": b, "weight": int(w)}
                for (a, b), w in kg_edges.most_common(500)
            ],
            "byte_histogram": byte_hist,
            "pseudo_weights": pseudo_weights,
        }


def search_index(index_blob: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    tokenizer = ByteWordTokenizer()
    query_words = tokenizer.tokenize_words(query)
    if not query_words:
        return []

    inverted = index_blob.get("inverted_index", {})
    matched_ids: Optional[Set[int]] = None

    for word in query_words:
        ids = set(inverted.get(word, []))
        if matched_ids is None:
            matched_ids = ids
        else:
            matched_ids &= ids

    if not matched_ids:
        return []

    sample_map = {item["doc_id"]: item for item in index_blob.get("samples", [])}
    return [sample_map[idx] for idx in sorted(matched_ids) if idx in sample_map]
