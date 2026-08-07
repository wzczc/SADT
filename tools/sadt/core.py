"""Core SADT utilities.

This module contains the reusable pieces used by the public SADT pipeline:
generation tracing, Image-Attention object filtering, LLCC hallucination
detection, HARM visual masking, VEED-style visual evidence guidance, and
semantic matching between decoded visual tokens and target object words.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

try:
    import inflect
except ImportError:  # pragma: no cover - optional dependency
    inflect = None



# Predefined object-similarity groups used by semantic_matcher="predefined".
# can be constructed with reference to object vocabularies such as COCO.
DEFAULT_SIMILAR_WORDS = [
    ["chair", "seat"],
    # ......
]


@dataclass
class SADTConfig:
    image_attention_layers: Tuple[int, ...] = tuple(range(19, 26))
    image_attention_threshold: float = 0.0
    topk_patches: int = 3
    patches_to_mask_per_word: int = 2
    model_image_size: int = 336
    image_token_count: int = 576
    grid_size: int = 24
    veed_alpha: float = 0.0
    semantic_matcher: str = "predefined"
    wordnet_similarity_threshold: float = 0.9
    llm_judge_api_base: Optional[str] = None
    llm_judge_model: Optional[str] = None
    llm_judge_api_key_env: str = "OPENAI_API_KEY"
    llm_judge_timeout: float = 20.0
    excluded_words: Tuple[str, ...] = ("person",)  # Tokens decoded from person regions are often concrete details such as clothing, colors, or body-related attributes rather than the word person.
    similar_words: Tuple[Tuple[str, ...], ...] = tuple(tuple(x) for x in DEFAULT_SIMILAR_WORDS)


@dataclass
class SADTDetection:
    word: str
    token_indices: List[int]
    avg_image_attention: float
    is_object_token: bool
    is_hallucination: bool
    right_num: int
    top_patch_indices: List[int]
    visual_tokens_by_layer: Dict[int, List[str]]
    visual_token_ids_by_layer: Dict[int, List[int]]
    veed_token: Optional[str] = None
    veed_token_id: Optional[int] = None
    hallucination_type: Optional[str] = None


def _similar_dict(groups: Sequence[Sequence[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for group in groups:
        for word in group:
            out.setdefault(word, [])
            out[word] = sorted(set(out[word] + [x for x in group if x != word]))
    return out


def _plural(word: str) -> Optional[str]:
    if not inflect:
        return word + "s" if not word.endswith("s") else word
    engine = inflect.engine()
    return engine.plural_noun(word) or engine.plural(word)


def _norm_token(text: str) -> str:
    text = text.lower().replace("\u2581", " ").replace("\u0120", " ")
    return re.sub(r"[^a-z0-9 ]+", "", text).strip()


def _wordnet_pos_forms(word: str) -> List[str]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    forms = []
    for synset in wn.synsets(word, pos=wn.NOUN):
        forms.extend(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    return forms


@lru_cache(maxsize=4096)
def _wordnet_synset_forms_cached(word: str) -> Tuple[str, ...]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return tuple()

    forms = set()
    for synset in wn.synsets(word, pos=wn.NOUN):
        forms.update(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    return tuple(sorted(_norm_token(item) for item in forms if _norm_token(item)))


def _wordnet_synset_forms(word: str) -> List[str]:
    try:
        return list(_wordnet_synset_forms_cached(_norm_token(word)))
    except LookupError:
        return []


@lru_cache(maxsize=4096)
def _wordnet_similar_forms_cached(word: str, threshold: float) -> Tuple[str, ...]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return tuple()

    target_synsets = wn.synsets(word, pos=wn.NOUN)
    if not target_synsets:
        return tuple()

    forms = set(_wordnet_pos_forms(word))
    for synset in wn.all_synsets(pos=wn.NOUN):
        score = 0.0
        for target in target_synsets:
            sim = synset.wup_similarity(target) or synset.path_similarity(target) or 0.0
            score = max(score, float(sim))
        if score >= threshold:
            forms.update(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    return tuple(sorted(_norm_token(item) for item in forms if _norm_token(item)))


def _wordnet_similar_forms(word: str, threshold: float) -> List[str]:
    try:
        return list(_wordnet_similar_forms_cached(word, float(threshold)))
    except LookupError:
        return []


@lru_cache(maxsize=1)
def _coco_synonym_groups() -> Tuple[Tuple[str, ...], ...]:
    text = ""
    chair_path = Path(__file__).resolve().parents[2] / "evaluation" / "chair.py"
    try:
        source = chair_path.read_text(encoding="utf-8")
    except OSError:
        source = ""
    match = re.search(r"synonyms_txt\s*=\s*'''(.*?)'''", source, flags=re.S)
    if match:
        text = match.group(1)
    else:
        try:
            from evaluation.chair import synonyms_txt

            text = synonyms_txt
        except Exception:
            text = ""

    groups = []
    for line in text.splitlines():
        items = [_norm_token(item) for item in line.strip().split(",")]
        items = sorted({item for item in items if item})
        if items:
            groups.append(tuple(items))
    return tuple(groups)


@lru_cache(maxsize=4096)
def _coco_synonym_forms_cached(word: str) -> Tuple[str, ...]:
    normalized = _norm_token(word)
    for group in _coco_synonym_groups():
        if normalized in group:
            return group
    return tuple()


def _coco_synonym_forms(word: str) -> List[str]:
    return list(_coco_synonym_forms_cached(_norm_token(word)))


def build_semantic_forms(word: str, config: SADTConfig) -> List[str]:
    mode = config.semantic_matcher.replace("_", "-").lower()
    mode_parts = set(mode.split("-"))
    similar = []
    normalized_word = _norm_token(word)
    if "predefined" in mode_parts:
        similar.extend(_similar_dict(config.similar_words).get(normalized_word, []))
    if "coco" in mode_parts:
        similar.extend(_coco_synonym_forms(normalized_word))
    if "wordnet" in mode_parts and "similarity" in mode_parts:
        similar.extend(_wordnet_similar_forms(word, config.wordnet_similarity_threshold))
    elif "wordnet" in mode_parts:
        similar.extend(_wordnet_synset_forms(normalized_word))
    forms = [word, _plural(word)] + similar + [_plural(item) for item in similar]
    return sorted({_norm_token(item) for item in forms if item})


def _uses_llm_judge(config: SADTConfig) -> bool:
    mode = config.semantic_matcher.replace("_", "-").lower()
    return "llm" in mode.split("-") and "judge" in mode.split("-")


@lru_cache(maxsize=65536)
def _llm_judge_semantic_consistency_cached(
    target_word: str,
    decoded_token: str,
    api_base: str,
    model: str,
    api_key_env: str,
    timeout: float,
) -> bool:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("LLM judge requires the 'requests' package.") from exc

    api_key = os.environ.get(api_key_env, "")
    if "api.openai.com" in api_base and not api_key:
        raise RuntimeError(
            f"LLM judge API key is missing. Set {api_key_env}, or pass "
            "--llm-judge-api-base for a local OpenAI-compatible endpoint."
        )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You judge whether a decoded visual token provides semantic visual evidence "
                    "for a target object category. "
                    "Answer only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target object: {target_word}\n"
                    f"Decoded visual token: {decoded_token}\n\n"
                    "Answer YES if the decoded token is: "
                    "1. the same object category as the target; "
                    "2. a synonym, plural/singular form, abbreviation, or common name of the target; "
                    "3. one meaningful word from a multi-word target object that still identifies "
                    "the object, such as 'phone' for 'cell phone' or 'racket' for 'tennis racket'; "
                    "4. a distinctive visible part or component that strongly indicates the target "
                    "object, such as 'screen' for 'laptop' or 'wheel' for 'bicycle'. "
                    "Answer NO if the decoded token is: "
                    "1. a different object; "
                    "2. only a generic part that is not specific to the target, such as 'leg', "
                    "'handle', 'top', or 'side'; "
                    "3. an attribute, color, material, action, scene/context word, or object that "
                    "merely co-occurs with the target; "
                    "4. only loosely related to the target."
                ),
            },
        ],
    }
    response = requests.post(api_base, headers=headers, data=json.dumps(payload), timeout=float(timeout))
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"].strip().lower()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM judge response: {data}") from exc
    if content.startswith("yes"):
        return True
    if content.startswith("no"):
        return False
    raise RuntimeError(f"Unexpected LLM judge answer: {content!r}")


def llm_judge_semantic_consistency(target_word: str, decoded_token: str, config: SADTConfig) -> bool:
    api_base = config.llm_judge_api_base or os.environ.get(
        "SADT_LLM_JUDGE_API_BASE",
        "https://api.openai.com/v1/chat/completions",
    )
    model = config.llm_judge_model or os.environ.get("SADT_LLM_JUDGE_MODEL", "gpt-4o-mini")
    return _llm_judge_semantic_consistency_cached(
        _norm_token(target_word),
        _norm_token(decoded_token),
        api_base,
        model,
        config.llm_judge_api_key_env,
        float(config.llm_judge_timeout),
    )


def is_semantically_consistent(target_word: str, decoded_token: str, config: SADTConfig) -> bool:
    token = _norm_token(decoded_token)
    if not token:
        return False
    mode = config.semantic_matcher.replace("_", "-").lower()
    if mode == "llm-judge":
        return llm_judge_semantic_consistency(target_word, token, config)
    forms = build_semantic_forms(target_word, config)
    if any(token == form or token in form or form in token for form in forms):
        return True
    if _uses_llm_judge(config):
        return llm_judge_semantic_consistency(target_word, token, config)
    return False


def find_token_spans(token_ids: Sequence[int], generated_ids: Sequence[int]) -> List[int]:
    n = len(token_ids)
    if n == 0:
        return []
    out = []
    for i in range(0, len(generated_ids) - n + 1):
        if list(generated_ids[i : i + n]) == list(token_ids):
            out.append(i)
    return out


def expand2square(pil_img: Image.Image, background_color: Tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    size = max(width, height)
    result = Image.new(pil_img.mode, (size, size), background_color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def _response_ids(output: Any, input_ids: torch.Tensor) -> List[int]:
    seq = output["sequences"][0] if isinstance(output, dict) else output.sequences[0]
    seq_list = seq.detach().cpu().tolist()
    input_len = int(input_ids.shape[1])
    if len(seq_list) > input_len and seq_list[:input_len] == input_ids[0].detach().cpu().tolist():
        return seq_list[input_len:]
    return seq_list[1:] if seq_list else []


def generate_with_trace(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    model_name: str,
    image_file: str,
    prompt: str,
    *,
    image_tensor: Optional[torch.Tensor] = None,
    image_sizes: Optional[List[Tuple[int, int]]] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 256,
    conv_mode: Optional[str] = None,
    logits_processor: Optional[Any] = None,
    hallu_indices: Optional[Sequence[int]] = None,
    hallu_logits: Optional[Sequence[torch.Tensor]] = None,
    cd_alpha: Optional[float] = None,
) -> Tuple[Any, torch.Tensor, str]:
    from llava.eval.run_llava import eval_model
    from llava.mm_utils import process_images

    if image_tensor is None:
        image = Image.open(image_file).convert("RGB")
        image_sizes = [image.size]
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    else:
        images_tensor = [image_tensor]
    if image_sizes is None:
        image_sizes = [Image.open(image_file).size]

    args = SimpleNamespace(
        model_base=None,
        model_path="",
        model_name=model_name,
        query=prompt,
        conv_mode=conv_mode,
        image_file=image_file,
        image_sizes=image_sizes,
        images_tensor=images_tensor,
        sep=",",
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        logits_processor=logits_processor,
        hallu_indices=list(hallu_indices) if hallu_indices is not None else None,
        hallu_logits=list(hallu_logits) if hallu_logits is not None else None,
        cd_alpha=cd_alpha,
    )
    output, input_ids = eval_model(args)
    text = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0].strip()
    return output, input_ids, text


def _attention_for_token(
    attentions: Sequence[Any],
    token_idx: int,
    img_idx: int,
    image_token_count: int,
) -> np.ndarray:
    attention_layers = torch.stack(attentions[token_idx])
    return (
        attention_layers[:, 0, :, -1, img_idx : img_idx + image_token_count]
        .detach()
        .float()
        .cpu()
        .numpy()
        .mean(axis=1)
    )


def _top_patches_across_layers(attention_img: np.ndarray, layers: Sequence[int], topk: int) -> List[int]:
    target_att = attention_img[list(layers)]
    token_sum_att = target_att.sum(axis=0)
    k = min(topk, token_sum_att.size)
    indices = np.argsort(token_sum_att)[::-1][:k]
    return [int(x) for x in indices]


def filter_words_by_image_attention(
    *,
    words: Sequence[str],
    output: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    config: SADTConfig,
) -> Dict[str, Any]:
    """Filter CHAIR-generated words by Image-Attention Stage attention.

    This implements the object-token identification step before LLCC. Words
    below ``image_attention_threshold`` are not sent to Logit-Lens detection.
    """
    attentions = output["attentions"]
    generated_ids = _response_ids(output, input_ids)
    img_idx = int(torch.nonzero(input_ids[0] == -200).squeeze().item())
    object_words: List[str] = []
    filtered_words: List[str] = []
    missing_words: List[str] = []
    attention_scores: Dict[str, float] = {}
    token_indices: Dict[str, List[int]] = {}

    for raw_word in words:
        word = raw_word.lower().strip()
        if not word:
            continue
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        indices = find_token_spans(ids, generated_ids)
        if not indices:
            plural = _plural(word)
            ids = tokenizer(plural, add_special_tokens=False)["input_ids"] if plural else ids
            indices = find_token_spans(ids, generated_ids)
        if not indices:
            missing_words.append(word)
            continue

        attention_img = _attention_for_token(
            attentions,
            indices[0],
            img_idx,
            config.image_token_count,
        )
        avg_attention = float(attention_img[list(config.image_attention_layers)].sum(axis=1).mean())
        attention_scores[word] = avg_attention
        token_indices[word] = indices
        if avg_attention > config.image_attention_threshold:
            object_words.append(word)
        else:
            filtered_words.append(word)

    return {
        "object_words": sorted(set(object_words)),
        "filtered_words": sorted(set(filtered_words)),
        "missing_words": sorted(set(missing_words)),
        "attention_scores": attention_scores,
        "token_indices": token_indices,
    }


def detect_hallucinations(
    *,
    words: Sequence[str],
    output: Any,
    input_ids: torch.Tensor,
    model: Any,
    tokenizer: Any,
    config: SADTConfig,
) -> List[SADTDetection]:
    attentions = output["attentions"]
    hidden_states = output["hidden_states"][0]
    generated_ids = _response_ids(output, input_ids)
    img_idx = int(torch.nonzero(input_ids[0] == -200).squeeze().item())
    detections: List[SADTDetection] = []

    for raw_word in words:
        word = raw_word.lower().strip()
        if not word or word in config.excluded_words:
            continue
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        token_indices = find_token_spans(token_ids, generated_ids)
        if not token_indices:
            plural = _plural(word)
            token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"] if plural else token_ids
            token_indices = find_token_spans(token_ids, generated_ids)
        if not token_indices:
            continue

        token_idx = token_indices[0]
        attention_img = _attention_for_token(attentions, token_idx, img_idx, config.image_token_count)
        avg_attention = float(attention_img[list(config.image_attention_layers)].sum(axis=1).mean())
        is_object = avg_attention > config.image_attention_threshold
        top_patch_indices = _top_patches_across_layers(
            attention_img,
            config.image_attention_layers,
            config.topk_patches,
        )

        visual_tokens_by_layer: Dict[int, List[str]] = {}
        visual_token_ids_by_layer: Dict[int, List[int]] = {}
        right_num = 0
        veed_token = None
        veed_token_id = None
        best_patch_idx = top_patch_indices[0] if top_patch_indices else None

        with torch.no_grad():
            for layer in config.image_attention_layers:
                layer_att = attention_img[layer]
                k = min(config.topk_patches, layer_att.size)
                layer_patch_indices = np.argsort(layer_att)[::-1][:k]
                absolute_indices = torch.as_tensor(layer_patch_indices + img_idx, device=model.device)
                layer_index = max(0, min(layer, len(hidden_states) - 1))
                visual_hidden = hidden_states[layer_index][:, absolute_indices, :].to(model.device)
                visual_logits = model.lm_head(visual_hidden)
                token_ids_tensor = torch.argmax(visual_logits, dim=-1).squeeze(0)
                token_ids_list = [int(x) for x in token_ids_tensor.detach().cpu().tolist()]
                visual_tokens = [
                    tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids_list
                ]
                visual_tokens_by_layer[int(layer)] = visual_tokens
                visual_token_ids_by_layer[int(layer)] = token_ids_list
                right_num += sum(
                    is_semantically_consistent(word, token, config) for token in visual_tokens
                )

                if best_patch_idx is not None and int(best_patch_idx) in [int(x) for x in layer_patch_indices]:
                    local = [int(x) for x in layer_patch_indices].index(int(best_patch_idx))
                    veed_token = visual_tokens[local]
                    veed_token_id = token_ids_list[local]

        detections.append(
            SADTDetection(
                word=word,
                token_indices=token_indices,
                avg_image_attention=avg_attention,
                is_object_token=is_object,
                is_hallucination=is_object and right_num == 0,
                right_num=right_num,
                top_patch_indices=top_patch_indices,
                visual_tokens_by_layer=visual_tokens_by_layer,
                visual_token_ids_by_layer=visual_token_ids_by_layer,
                veed_token=veed_token,
                veed_token_id=veed_token_id,
            )
        )
    return detections


def mask_high_attention_regions_vcd(
    image_file: str,
    detections: Sequence[SADTDetection],
    config: SADTConfig,
    image_mean: Optional[Sequence[float]] = None,
) -> Optional[Image.Image]:
    """Return an in-memory HARM image using the original VCD patch mapping."""
    patch_indices: List[int] = []
    for item in detections:
        if item.is_hallucination:
            patch_indices.extend(item.top_patch_indices[: config.patches_to_mask_per_word])
    if not patch_indices:
        return None

    image = Image.open(image_file).convert("RGB")
    image_size = image.size
    background = tuple(int(x * 255) for x in image_mean) if image_mean else (0, 0, 0)
    image = expand2square(image, background)
    img_np = np.array(image).copy()

    patch_nums = config.model_image_size / (config.model_image_size / config.grid_size)
    patch_size = (config.model_image_size / config.grid_size) * image_size[0] / config.model_image_size

    color = (0, 0, 0)

    for token_idx in sorted(set(int(x) for x in patch_indices)):
        row = token_idx // patch_nums
        col = token_idx % patch_nums
        y0 = int(row * patch_size)
        y1 = int(y0 + patch_size)
        x0 = int(col * patch_size)
        x1 = int(x0 + patch_size)
        img_np[y0:y1, x0:x1, :] = color

    return Image.fromarray(img_np)


def _first_token_id_for_generated_word(tokenizer: Any, word: str, generated_ids: Sequence[int]) -> Optional[int]:
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if ids and find_token_spans(ids, generated_ids):
        return int(ids[0])

    plural_word = _plural(word)
    if plural_word:
        ids = tokenizer(plural_word, add_special_tokens=False)["input_ids"]
        if ids:
            return int(ids[0])
    return None


def build_vcd_hallucination_guidance(
    *,
    detections: Sequence[SADTDetection],
    output: Any,
    input_ids: torch.Tensor,
    model: Any,
    tokenizer: Any,
    config: SADTConfig,
    hidden_layer_index: int = -8,
) -> Tuple[List[int], List[torch.Tensor]]:
    """Build the original VCD-style hallucination token ids and visual logits.

    This mirrors ``VCD/ours.py``: for each Type-2 hallucination, take the first
    high-attention visual patch, project its visual hidden state from layer
    ``-8`` with the LM head, and use the generated word's first token id as the
    decoding trigger.
    """
    contextual = [
        item
        for item in detections
        if item.is_hallucination and item.hallucination_type == "contextual_prior" and item.top_patch_indices
    ]
    if not contextual:
        return [], []

    hidden_states = output["hidden_states"][0]
    generated_ids = _response_ids(output, input_ids)
    img_idx = int(torch.nonzero(input_ids[0] == -200).squeeze().item())
    trigger_token_ids: List[int] = []
    evidence_logits: List[torch.Tensor] = []
    layer_index = hidden_layer_index
    if layer_index < 0:
        layer_index = len(hidden_states) + layer_index
    layer_index = max(0, min(layer_index, len(hidden_states) - 1))

    with torch.no_grad():
        for item in contextual:
            patch_idx = int(item.top_patch_indices[0])
            visual_hidden = hidden_states[layer_index][:, img_idx + patch_idx, :].to(model.device)
            visual_logits = model.lm_head(visual_hidden).detach()
            token_id = _first_token_id_for_generated_word(tokenizer, item.word, generated_ids)
            if token_id is None:
                continue
            trigger_token_ids.append(token_id)
            evidence_logits.append(visual_logits)

    return trigger_token_ids, evidence_logits
