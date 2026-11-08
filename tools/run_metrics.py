"""Run artifact metric parsers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from preprocess.jsonl import JsonValue, json_texts, read_jsonl


def _json_file(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return cast(dict[str, JsonValue], value)
    return {}


def _int(value: JsonValue) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _list_len(value: JsonValue) -> int:
    return len(value) if isinstance(value, list) else 0


def _length_stats(values: list[int]) -> dict[str, JsonValue]:
    return {
        "min": min(values) if values else 0,
        "mean": round(sum(values) / len(values), 2) if values else 0.0,
        "max": max(values) if values else 0,
    }


def _number_stats(values: list[float]) -> dict[str, JsonValue]:
    return {
        "min": round(min(values), 6) if values else 0.0,
        "mean": round(sum(values) / len(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
    }


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * fraction))),
    )
    return float(ordered[index])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    kinds: Counter[str] = Counter()
    tokens: list[int] = []
    documents: set[str] = set()
    sections: set[str] = set()
    texts: set[str] = set()
    duplicates = 0
    for row in read_jsonl(path):
        kinds[str(row.get("kind", ""))] += 1
        tokens.append(_int(row.get("token_count", 0)))
        documents.add(str(row.get("doc_id", "")))
        sections.add(str(row.get("section", "")))
        text = str(row.get("text", ""))
        if text in texts:
            duplicates += 1
        texts.add(text)
    return {
        "chunks": len(tokens),
        "kinds": dict(sorted(kinds.items())),
        "documents": len(documents),
        "sections": len(sections),
        "duplicate_texts": duplicates,
        "tokens": {
            "min": min(tokens) if tokens else 0,
            "p50": _percentile(tokens, 0.50),
            "p95": _percentile(tokens, 0.95),
            "p99": _percentile(tokens, 0.99),
            "max": max(tokens) if tokens else 0,
            "total": sum(tokens),
        },
    }


def _golden_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    examples = 0
    questions: set[str] = set()
    duplicates = 0
    documents: set[str] = set()
    terms: list[int] = []
    question_chars: list[int] = []
    answer_chars: list[int] = []
    for row in read_jsonl(path):
        examples += 1
        expected = json_texts(row.get("expected_terms", []))
        terms.append(len(expected))
        answer_chars.append(len(expected[0]) if expected else 0)
        question = str(row.get("question", ""))
        question_chars.append(len(question))
        if question in questions:
            duplicates += 1
        questions.add(question)
        documents.update(json_texts(row.get("expected_doc_ids", [])))
    return {
        "examples": examples,
        "duplicate_questions": duplicates,
        "documents": len(documents),
        "expected_terms": _length_stats(terms),
        "question_chars": _length_stats(question_chars),
        "answer_chars": _length_stats(answer_chars),
    }


def _fact_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    facts = 0
    empty = 0
    grounded = 0
    documents: set[str] = set()
    for row in read_jsonl(path):
        facts += 1
        if not str(row.get("text", "")).strip():
            empty += 1
        if row.get("grounded") is True:
            grounded += 1
        documents.add(str(row.get("doc_id", "")))
    return {
        "facts": facts,
        "empty": empty,
        "grounded": grounded,
        "ungrounded": facts - grounded,
        "documents": len(documents),
    }


def _concept_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    concepts = 0
    empty = 0
    aliases = 0
    linked_chunks = 0
    linked_facts = 0
    documents: set[str] = set()
    for row in read_jsonl(path):
        concepts += 1
        if not str(row.get("definition", "")).strip():
            empty += 1
        aliases += _list_len(row.get("aliases", []))
        linked_chunks += _list_len(row.get("chunk_ids", []))
        linked_facts += _list_len(row.get("fact_ids", []))
        documents.add(str(row.get("doc_id", "")))
    return {
        "concepts": concepts,
        "empty_definitions": empty,
        "aliases": aliases,
        "linked_chunks": linked_chunks,
        "linked_facts": linked_facts,
        "documents": len(documents),
    }


def _raft_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    examples = 0
    absent = 0
    statuses: Counter[str] = Counter()
    contexts: Counter[int] = Counter()
    questions: set[str] = set()
    duplicates = 0
    question_chars: list[int] = []
    answer_chars: list[int] = []
    weights: list[float] = []
    supported = 0
    absent_verified = 0
    for row in read_jsonl(path):
        examples += 1
        status = str(row.get("status", ""))
        statuses[status] += 1
        absent += int(status == "absent")
        contexts[_list_len(row.get("contexts", []))] += 1
        question = str(row.get("question", ""))
        question_chars.append(len(question))
        if question in questions:
            duplicates += 1
        questions.add(question)
        answer_chars.append(len(str(row.get("answer", ""))))
        weight = row.get("weight", 1.0)
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            weights.append(float(weight))
        if row.get("oracle_supported") is True:
            supported += 1
        if row.get("absent_verified") is True:
            absent_verified += 1
    return {
        "examples": examples,
        "absent": absent,
        "statuses": dict(sorted(statuses.items())),
        "contexts": {str(key): value for key, value in sorted(contexts.items())},
        "duplicate_questions": duplicates,
        "oracle_supported": supported,
        "absent_verified": absent_verified,
        "weight": _number_stats(weights),
        "question_chars": _length_stats(question_chars),
        "answer_chars": _length_stats(answer_chars),
    }


def _behavior_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    result: dict[str, JsonValue] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("answer hits:"):
            hits, total = line.split(":", 1)[1].split("=", 1)[0].strip().split("/", 1)
            result["answer"] = {"hits": _int(hits), "total": _int(total)}
        elif line.startswith("citation support hits:"):
            hits, total = line.split(":", 1)[1].split("=", 1)[0].strip().split("/", 1)
            result["citation"] = {"hits": _int(hits), "total": _int(total)}
        elif line.startswith("refusals:"):
            hits, total = line.split(":", 1)[1].split("=", 1)[0].strip().split("/", 1)
            result["refusal"] = {"hits": _int(hits), "total": _int(total)}
        elif line.startswith("answerable refusals:"):
            hits, total = line.split(":", 1)[1].split("=", 1)[0].strip().split("/", 1)
            result["answerable_refusal"] = {
                "hits": _int(hits),
                "total": _int(total),
            }
        elif line.startswith("refusal leakage:"):
            hits, total = line.split(":", 1)[1].split("=", 1)[0].strip().split("/", 1)
            result["refusal_leakage"] = {"hits": _int(hits), "total": _int(total)}
    return result


def _retrieval_report(path: Path) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if " recall@" not in line or " = " not in line:
            continue
        label, rest = line.split(" recall@", 1)
        top_k_text, rest = rest.split(":", 1)
        hits_text, recall_text = rest.split("=", 1)
        hits, total = hits_text.strip().split("/", 1)
        result[label] = {
            "top_k": _int(top_k_text),
            "hits": _int(hits),
            "total": _int(total),
            "recall": float(recall_text),
        }
    return result


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return -1.0


def _gpu_report(path: Path) -> dict[str, JsonValue]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    try:
        use_index = header.index("GPU use (%)")
        vram_index = header.index("GPU Memory Allocated (VRAM%)")
        power_index = header.index("Average Graphics Package Power (W)")
    except ValueError:
        return {"samples": len(lines) - 1}
    values: dict[str, dict[str, list[float]]] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(use_index, vram_index, power_index):
            continue
        card = parts[1]
        bucket = values.setdefault(
            card,
            {"gpu_use": [], "vram": [], "power": []},
        )
        for key, index in (
            ("gpu_use", use_index),
            ("vram", vram_index),
            ("power", power_index),
        ):
            number = _float(parts[index])
            if number >= 0.0:
                bucket[key].append(number)
    cards: dict[str, JsonValue] = {}
    for card, bucket in sorted(values.items()):
        cards[card] = {
            "gpu_use_max": max(bucket["gpu_use"]) if bucket["gpu_use"] else 0.0,
            "gpu_use_mean": (
                sum(bucket["gpu_use"]) / len(bucket["gpu_use"])
                if bucket["gpu_use"]
                else 0.0
            ),
            "vram_max": max(bucket["vram"]) if bucket["vram"] else 0.0,
            "power_max": max(bucket["power"]) if bucket["power"] else 0.0,
            "samples": len(bucket["gpu_use"]),
        }
    return {"samples": len(lines) - 1, "cards": cards}


def _files(root: Path) -> list[dict[str, JsonValue]]:
    names = (
        "preprocess/docs.jsonl",
        "preprocess/facts.jsonl",
        "preprocess/concepts.jsonl",
        "preprocess/chunks.jsonl",
        "preprocess/audit.json",
        "preprocess/normalize_summary.json",
        "preprocess/structure_summary.json",
        "preprocess/rag.sqlite",
        "golden/golden.generated.jsonl",
        "golden/train.jsonl",
        "golden/holdout.jsonl",
        "golden/refusal_questions.jsonl",
        "raft/raft.jsonl",
        "train/adapter/summary.json",
        "train/adapter/history.jsonl",
        "train/adapter/adapter_model.safetensors",
    )
    result: list[dict[str, JsonValue]] = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        result.append(
            {
                "path": name,
                "bytes": size,
                "sha256": _sha256(path) if size <= 64 * 1024 * 1024 else "",
            }
        )
    return result
