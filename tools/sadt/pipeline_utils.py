"""Shared utilities for SADT command-line pipelines."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig,
    SADTDetection,
    expand2square,
    mask_high_attention_regions_vcd,
)


SEMANTIC_MATCHER_CHOICES = [
    "predefined",
    "wordnet",
    "wordnet-similarity",
    "coco",
    "coco-wordnet",
    "llm-judge",
]


def parse_layers(text: str) -> Tuple[int, ...]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return tuple(values)


def parse_words(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def image_sort_key(path: Path) -> Tuple[int, str]:
    digits = "".join(ch for ch in path.stem.split("_")[0] if ch.isdigit())
    return (int(digits) if digits else 10**18, path.name)


def collect_images(image_file: Optional[str], image_dir: Optional[str], limit: Optional[int]) -> List[str]:
    if image_file:
        return [image_file]
    if not image_dir:
        raise ValueError("Either --image-file or --image-dir is required.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [path for path in Path(image_dir).iterdir() if path.suffix.lower() in exts],
        key=image_sort_key,
    )
    if limit is not None:
        images = images[:limit]
    return [str(path) for path in images]


def load_labels(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data.values() if isinstance(data, dict) else data
    labels = {}
    for item in items:
        name = item.get("img_id") or item.get("image") or item.get("image_file")
        if name:
            labels[Path(name).name] = item
    return labels


def load_chair_evaluator(chair_cache: Optional[str], coco_annotations: Optional[str]):
    if chair_cache and Path(chair_cache).exists():
        from evaluation import chair as chair_module

        sys.modules.setdefault("chair", chair_module)
        with open(chair_cache, "rb") as handle:
            return pickle.load(handle)
    if coco_annotations:
        from evaluation.chair import CHAIR

        evaluator = CHAIR(coco_annotations)
        if chair_cache:
            Path(chair_cache).parent.mkdir(parents=True, exist_ok=True)
            with open(chair_cache, "wb") as handle:
                pickle.dump(evaluator, handle)
        return evaluator
    return None


def chair_words(evaluator: Any, image_name: str, response: str) -> Optional[Dict[str, List[str]]]:
    if evaluator is None:
        return None
    from evaluation.chair import chair_eval

    info = chair_eval(evaluator, image_name, response)
    hallucinated_words = []
    for item in info.get("mscoco_hallucinated_words", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            hallucinated_words.append(str(item[1]))
        else:
            hallucinated_words.append(str(item))
    return {
        "generated_words": sorted(set(info.get("mscoco_generated_words", []))),
        "gt_words": sorted(set(info.get("mscoco_gt_words", []))),
        "hallucination_words": sorted(set(word.lower().strip() for word in hallucinated_words if word.strip())),
    }


def choose_words(
    *,
    explicit_words: Optional[Sequence[str]],
    label_item: Optional[Dict[str, Any]],
    chair_item: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if explicit_words:
        generated_words = sorted(set(word.lower().strip() for word in explicit_words if word.strip()))
        source = "manual_candidate_objects"
    elif chair_item and chair_item.get("generated_words"):
        generated_words = sorted(set(word.lower().strip() for word in chair_item["generated_words"] if word.strip()))
        source = "chair_generated_words"
    else:
        raise ValueError(
            "generated_words must come from CHAIR to match the original code. "
            "Pass --chair-cache or --coco-annotations, or use --candidate-objects for manual debugging."
        )

    if label_item and label_item.get("gt_words"):
        gt_words = sorted(set(word.lower().strip() for word in label_item["gt_words"] if str(word).strip()))
        gt_source = "labels_json"
    elif chair_item and chair_item.get("gt_words"):
        gt_words = sorted(set(word.lower().strip() for word in chair_item["gt_words"] if str(word).strip()))
        gt_source = "chair_gt_words"
    else:
        gt_words = []
        gt_source = "none"

    return {
        "generated_words": generated_words,
        "generated_words_source": source,
        "gt_words": gt_words,
        "gt_words_source": gt_source,
    }


def _normalized_words(words: Optional[Sequence[Any]]) -> List[str]:
    if not words:
        return []
    return sorted(set(str(word).lower().strip() for word in words if str(word).strip()))


def _label_hallucination_words(label_item: Optional[Dict[str, Any]]) -> List[str]:
    if not label_item:
        return []
    if "hallucination_words" in label_item:
        return _normalized_words(label_item.get("hallucination_words"))
    if "mscoco_hallucinated_words" in label_item:
        words = []
        for item in label_item["mscoco_hallucinated_words"]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                words.append(item[1])
            else:
                words.append(item)
        return _normalized_words(words)
    return []


def label_hallucination_words(
    *,
    label_item: Optional[Dict[str, Any]],
    excluded_words: Sequence[str],
) -> Dict[str, Any]:
    """Read the ground-truth hallucination words from metadata labels."""
    excluded = set(excluded_words)
    label_words = [word for word in _label_hallucination_words(label_item) if word not in excluded]
    return {"hallucination_words": label_words, "hallucination_words_source": "labels_json"}


def save_json_output(path: str, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def prepare_vcd_image_tensor(
    image_source: str | Path | Image.Image,
    image_processor: Any,
    model: Any,
    image_sizes: Optional[List[Tuple[int, int]]] = None,
):
    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")
    if image_sizes is None:
        image_sizes = [image.size]
    background = tuple(int(x * 255) for x in image_processor.image_mean)
    image = expand2square(image, background)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(
        model.device,
        dtype=torch.float16,
    )
    return image_tensor, image_sizes


def chair_output_record(
    *,
    evaluator: Any,
    image_name: str,
    response: str,
    gt_words: Sequence[str],
    excluded_words: Sequence[str],
    generated_words_source: str = "chair_generated_words",
) -> Dict[str, Any]:
    item = chair_words(evaluator, image_name, response)
    if item is None:
        raise ValueError("CHAIR evaluator is required to report generated/hallucination words.")

    excluded = set(excluded_words)
    generated_words = sorted(set(item["generated_words"]))
    gt_set = set(gt_words)
    hallucination_words = [
        word for word in generated_words if word not in gt_set and word not in excluded
    ]
    return {
        "text": response,
        "generated_words_source": generated_words_source,
        "generated_words": generated_words,
        "hallucination_words": hallucination_words,
    }


def chair_summary(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    sent_count = len(records)
    generated_count = 0
    hallucination_count = 0
    hallucinated_sentences = 0
    for item in records:
        output = item.get(field) or {}
        generated = output.get("generated_words") or []
        hallucinated = output.get("hallucination_words") or []
        generated_count += len(set(generated))
        hallucination_count += len(set(hallucinated))
        hallucinated_sentences += int(bool(hallucinated))
    return {
        "CHAIRs": hallucinated_sentences / sent_count if sent_count else 0.0,
        "CHAIRi": hallucination_count / generated_count if generated_count else 0.0,
    }


def detection_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    for item in records:
        detection = item.get("detection") or {}
        if "hallucination_words" in detection:
            true_hallu = set(detection.get("hallucination_words") or [])
        else:
            gt_words = set(item.get("gt_words") or [])
            object_words = detection.get("object_words") or []
            true_hallu = {word for word in object_words if word not in gt_words}
        pred_hallu = set(detection.get("detected_hallucinations") or [])
        tp += len(true_hallu & pred_hallu)
        fp += len(pred_hallu - true_hallu)
        fn += len(true_hallu - pred_hallu)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    type1 = type2 = 0
    for item in records:
        classification = item.get("classification") or {}
        type1 += len(set(classification.get("type1_hallucinations") or []))
        type2 += len(set(classification.get("type2_hallucinations") or []))
    return {
        "type1_visual_uncertainty": type1,
        "type2_contextual_prior": type2,
    }


def set_detection_types_from_words(
    detections: Sequence[SADTDetection],
    *,
    type1_words: Sequence[str],
    type2_words: Sequence[str],
) -> None:
    type1 = set(type1_words)
    type2 = set(type2_words)
    for item in detections:
        if not item.is_hallucination:
            continue
        if item.word in type2:
            item.hallucination_type = "contextual_prior"
        elif item.word in type1:
            item.hallucination_type = "visual_uncertainty"


def mask_only_words(
    *,
    image_file: str,
    detections: Sequence[SADTDetection],
    words: Sequence[str],
    config: SADTConfig,
    image_mean: Any,
) -> Optional[Image.Image]:
    words = set(words)
    if not words:
        return None

    old_flags = [item.is_hallucination for item in detections]
    try:
        for item in detections:
            item.is_hallucination = item.word in words
        return mask_high_attention_regions_vcd(
            image_file,
            detections,
            config,
            image_mean,
        )
    finally:
        for item, old_flag in zip(detections, old_flags):
            item.is_hallucination = old_flag
