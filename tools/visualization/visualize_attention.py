"""Attention statistic utilities for SADT/LLaVA.

This script is a cleaned release version of the original experimental
``vis.py``. It collects attention statistics for generated object words,
hallucinated object words, real object words, and non-object tokens, then
saves the pickle file used by ``paper_figures.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inflect
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


ENGINE = inflect.engine()


@dataclass
class AttentionStats:
    """Container for aggregate attention statistics."""

    all_att_sys: list[list[float]]
    all_att_img: list[list[float]]
    all_att_text: list[list[float]]
    real_att_sys: list[list[float]]
    real_att_img: list[list[float]]
    real_att_text: list[list[float]]
    hallu_att_sys: list[list[float]]
    hallu_att_img: list[list[float]]
    hallu_att_text: list[list[float]]
    non_att_sys: list[list[float]]
    non_att_img: list[list[float]]
    non_att_text: list[list[float]]
    real_image_attention: list[float]
    hallu_image_attention: list[float]
    non_image_attention: list[float]

    @classmethod
    def empty(cls) -> "AttentionStats":
        return cls(
            all_att_sys=[],
            all_att_img=[],
            all_att_text=[],
            real_att_sys=[],
            real_att_img=[],
            real_att_text=[],
            hallu_att_sys=[],
            hallu_att_img=[],
            hallu_att_text=[],
            non_att_sys=[],
            non_att_img=[],
            non_att_text=[],
            real_image_attention=[],
            hallu_image_attention=[],
            non_image_attention=[],
        )

    def to_dict(self) -> dict[str, list]:
        return {
            "all_att_sys": self.all_att_sys,
            "all_att_img": self.all_att_img,
            "all_att_tex": self.all_att_text,
            "real_att_sys": self.real_att_sys,
            "real_att_img": self.real_att_img,
            "real_att_tex": self.real_att_text,
            "hallu_att_sys": self.hallu_att_sys,
            "hallu_att_img": self.hallu_att_img,
            "hallu_att_tex": self.hallu_att_text,
            "non_att_sys": self.non_att_sys,
            "non_att_img": self.non_att_img,
            "non_att_tex": self.non_att_text,
            "all_real_attentions": self.real_image_attention,
            "all_hallu_attentions": self.hallu_image_attention,
            "all_non_attentions": self.non_image_attention,
        }


def parse_layer_range(value: str) -> range:
    """Parse a Python-like half-open layer range, for example ``19:26``."""

    start_text, end_text = value.split(":", maxsplit=1)
    return range(int(start_text), int(end_text))


def infer_num_layers(model_path: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return fallback
    lowered = model_path.lower()
    if "13b" in lowered:
        return 40
    if "7b" in lowered:
        return 32
    raise ValueError("Please pass --num-layers for models that are not 7B/13B.")


def find_token_sequence_indices(target_ids: Sequence[int], generated_ids: Sequence[int]) -> list[int]:
    """Find every start index where ``target_ids`` appears in ``generated_ids``."""

    target_len = len(target_ids)
    if target_len == 0:
        return []
    if target_len == 1:
        target = target_ids[0]
        return [idx for idx, token_id in enumerate(generated_ids) if token_id == target]
    return [
        idx
        for idx in range(len(generated_ids) - target_len + 1)
        if list(generated_ids[idx : idx + target_len]) == list(target_ids)
    ]


def locate_words_in_generation(tokenizer, words: Iterable[str], generated_ids: torch.Tensor) -> list[list[int]]:
    """Map object words to generated-token positions, with plural fallback."""

    generated_list = generated_ids[1:].tolist()
    word_indices = []
    for word in words:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        indices = find_token_sequence_indices(token_ids, generated_list)
        if not indices:
            plural = ENGINE.plural(word)
            token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"]
            indices = find_token_sequence_indices(token_ids, generated_list)
        word_indices.append(indices)
    return word_indices


def expand_indices(indices: Iterable[Iterable[int]], offsets: Sequence[int]) -> set[int]:
    expanded = set()
    for group in indices:
        for idx in group:
            expanded.add(idx)
            for offset in offsets:
                expanded.add(idx + offset)
    return expanded


def split_attention_by_prompt_region(
    attention_layers: torch.Tensor,
    img_idx: int,
    image_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return attention to system, image, and text regions for each layer."""

    attention_sys = attention_layers[:, 0, :, -1, :img_idx].mean(1)
    attention_img = attention_layers[:, 0, :, -1, img_idx : img_idx + image_token_count].mean(1)
    attention_text = attention_layers[:, 0, :, -1, img_idx + image_token_count :].mean(1)
    return attention_sys, attention_img, attention_text


def normalized_region_attention_across_layers(
    token_indices: Iterable[Iterable[int]],
    attentions,
    img_idx: int,
    num_layers: int,
    image_token_count: int,
) -> list[tuple[list[float], list[float], list[float]]]:
    """Compute per-layer normalized attention ratios for selected tokens."""

    word_attentions = []
    for indices in token_indices:
        if not indices:
            continue
        token_idx = indices[0]
        attention_layers = torch.stack(attentions[token_idx])
        attention_sys, attention_img, attention_text = split_attention_by_prompt_region(
            attention_layers,
            img_idx,
            image_token_count,
        )

        sys_values, img_values, text_values = [], [], []
        for layer_idx in range(num_layers):
            sys_sum = torch.sum(attention_sys[layer_idx])
            img_sum = torch.sum(attention_img[layer_idx])
            text_sum = torch.sum(attention_text[layer_idx])
            total = sys_sum + img_sum + text_sum
            sys_values.append((sys_sum / total).item())
            img_values.append((img_sum / total).item())
            text_values.append((text_sum / total).item())
        word_attentions.append((sys_values, img_values, text_values))
    return word_attentions


def non_object_attention_across_layers(
    object_token_indices: Iterable[Iterable[int]],
    attentions,
    img_idx: int,
    num_layers: int,
    image_token_count: int,
) -> list[tuple[list[float], list[float], list[float]]]:
    excluded = expand_indices(object_token_indices, offsets=(-1, 1))
    selected_indices = [[idx] for idx in range(len(attentions)) if idx not in excluded]
    return normalized_region_attention_across_layers(
        selected_indices,
        attentions,
        img_idx,
        num_layers,
        image_token_count,
    )


def image_attention_ratio_for_layers(
    token_indices: Iterable[Iterable[int]],
    attentions,
    img_idx: int,
    target_layers: range,
    image_token_count: int,
) -> list[float]:
    """Compute image-attention ratio over selected layers for selected tokens."""

    ratios = []
    for indices in token_indices:
        if not indices:
            continue
        token_idx = indices[0]
        attention_layers = torch.stack(attentions[token_idx])
        attention_sys, attention_img, attention_text = split_attention_by_prompt_region(
            attention_layers[list(target_layers)],
            img_idx,
            image_token_count,
        )

        img_sum = attention_img.sum().item()
        sys_sum = attention_sys.sum().item()
        text_sum = attention_text.sum().item()
        ratios.append(img_sum / (img_sum + sys_sum + text_sum))
    return ratios


def non_object_image_attention_ratio_for_layers(
    object_token_indices: Iterable[Iterable[int]],
    attentions,
    img_idx: int,
    target_layers: range,
    image_token_count: int,
) -> list[float]:
    excluded = expand_indices(object_token_indices, offsets=(-3, -2, -1, 1, 2, 3))
    selected_indices = [[idx] for idx in range(len(attentions)) if idx not in excluded]
    return image_attention_ratio_for_layers(
        selected_indices,
        attentions,
        img_idx,
        target_layers,
        image_token_count,
    )


def append_layer_attention(
    destination_sys: list[list[float]],
    destination_img: list[list[float]],
    destination_text: list[list[float]],
    values: Iterable[tuple[list[float], list[float], list[float]]],
) -> None:
    for sys_values, img_values, text_values in values:
        destination_sys.append(sys_values)
        destination_img.append(img_values)
        destination_text.append(text_values)


def plot_attention_across_layers(
    all_att_sys: Sequence[Sequence[float]],
    all_att_img: Sequence[Sequence[float]],
    all_att_text: Sequence[Sequence[float]],
    num_layers: int,
    save_path: Path,
) -> None:
    """Plot average attention ratio to system/image/text across layers."""

    if not all_att_sys:
        return

    avg_att_sys = np.array(all_att_sys).mean(axis=0)
    avg_att_img = np.array(all_att_img).mean(axis=0)
    avg_att_text = np.array(all_att_text).mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    layers = np.arange(1, num_layers + 1)
    ax.plot(layers, avg_att_sys, label="System", color="#1f77b4", linewidth=2.0)
    ax.plot(layers, avg_att_img, label="Image", color="#ff7f0e", linewidth=2.0)
    ax.plot(layers, avg_att_text, label="Text", color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel("Average Attention Ratio", fontweight="bold")
    ax.set_xticks(range(1, num_layers + 1, 4))
    ax.set_xlim(1, num_layers)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="lightgray")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_comparison(stats: AttentionStats, save_path: Path) -> None:
    groups = [
        ("Real Object", stats.real_image_attention),
        ("Hallucinated Object", stats.hallu_image_attention),
        ("Non-object Token", stats.non_image_attention),
    ]
    groups = [(name, values) for name, values in groups if values]
    if not groups:
        return

    labels = [name for name, _ in groups]
    means = [float(np.mean(values)) for _, values in groups]
    stds = [float(np.std(values)) for _, values in groups]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=["#4C78A8", "#E45756", "#54A24B"], alpha=0.78)
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Image Attention Ratio")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_chair_evaluator(cache_path: Path | None, annotation_dir: Path | None):
    from evaluation import chair as chair_module
    from evaluation.chair import CHAIR

    sys.modules.setdefault("chair", chair_module)

    if cache_path and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    if annotation_dir is None:
        raise ValueError("Pass --chair-cache or --coco-annotations to run CHAIR evaluation.")

    evaluator = CHAIR(str(annotation_dir))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(evaluator, handle)
    return evaluator


def sorted_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        return image_paths[:limit]
    return image_paths


def collect_statistics(args: argparse.Namespace) -> AttentionStats:
    if args.repo_root:
        sys.path.insert(0, str(Path(args.repo_root).resolve()))

    from evaluation.chair import chair_eval
    from llava.eval.run_llava import disable_torch_init, eval_model_ori
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    labels = None
    if args.labels_json:
        with Path(args.labels_json).open("r", encoding="utf-8") as handle:
            labels = json.load(handle)

    evaluator = load_chair_evaluator(
        Path(args.chair_cache) if args.chair_cache else None,
        Path(args.coco_annotations) if args.coco_annotations else None,
    )

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=get_model_name_from_path(args.model_path),
    )

    num_layers = infer_num_layers(args.model_path, args.num_layers)
    target_layers = parse_layer_range(args.target_layers)
    stats = AttentionStats.empty()
    image_paths = sorted_images(Path(args.image_dir), args.limit)
    excluded_words = {"car", "person"}

    for image_idx, image_path in enumerate(tqdm(image_paths, desc="Collecting attention")):
        run_args = type(
            "Args",
            (),
            {
                "model_base": args.model_base,
                "model_path": args.model_path,
                "model_name": get_model_name_from_path(args.model_path),
                "query": args.prompt,
                "conv_mode": args.conv_mode,
                "image_file": str(image_path),
                "sep": ",",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "model": model,
                "tokenizer": tokenizer,
                "image_processor": image_processor,
                "output_attentions": True,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            },
        )()

        output, input_ids = eval_model_ori(run_args)
        generated_ids = output["sequences"][0]
        answer = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0]
        img_idx = torch.nonzero(input_ids[0] == -200).squeeze().item()
        attentions = output["attentions"]

        image_info = chair_eval(evaluator, image_path.name, answer)
        label_item = None
        if isinstance(labels, dict):
            label_item = (
                labels.get(image_path.name)
                or labels.get(str(image_path))
                or labels.get(image_path.stem)
            )
        elif labels and image_idx < len(labels):
            label_item = labels[image_idx]
        gt_words = set(label_item.get("gt_words", [])) if label_item else set(image_info["mscoco_gt_words"])
        generated_words = set(image_info["mscoco_generated_words"])
        real_words = sorted(word for word in generated_words if word in gt_words)
        hallu_words = sorted(word for word in generated_words if word not in gt_words and word not in excluded_words)
        generated_words = sorted(generated_words)

        real_token_indices = locate_words_in_generation(tokenizer, real_words, generated_ids)
        hallu_token_indices = locate_words_in_generation(tokenizer, hallu_words, generated_ids)
        generated_token_indices = locate_words_in_generation(tokenizer, generated_words, generated_ids)

        stats.real_image_attention.extend(
            image_attention_ratio_for_layers(
                real_token_indices,
                attentions,
                img_idx,
                target_layers,
                args.image_token_count,
            )
        )
        stats.hallu_image_attention.extend(
            image_attention_ratio_for_layers(
                hallu_token_indices,
                attentions,
                img_idx,
                target_layers,
                args.image_token_count,
            )
        )
        stats.non_image_attention.extend(
            non_object_image_attention_ratio_for_layers(
                generated_token_indices,
                attentions,
                img_idx,
                target_layers,
                args.image_token_count,
            )
        )

        append_layer_attention(
            stats.all_att_sys,
            stats.all_att_img,
            stats.all_att_text,
            normalized_region_attention_across_layers(
                generated_token_indices,
                attentions,
                img_idx,
                num_layers,
                args.image_token_count,
            ),
        )
        append_layer_attention(
            stats.real_att_sys,
            stats.real_att_img,
            stats.real_att_text,
            normalized_region_attention_across_layers(
                real_token_indices,
                attentions,
                img_idx,
                num_layers,
                args.image_token_count,
            ),
        )
        append_layer_attention(
            stats.hallu_att_sys,
            stats.hallu_att_img,
            stats.hallu_att_text,
            normalized_region_attention_across_layers(
                hallu_token_indices,
                attentions,
                img_idx,
                num_layers,
                args.image_token_count,
            ),
        )
        append_layer_attention(
            stats.non_att_sys,
            stats.non_att_img,
            stats.non_att_text,
            non_object_attention_across_layers(
                generated_token_indices,
                attentions,
                img_idx,
                num_layers,
                args.image_token_count,
            ),
        )

    return stats


def save_outputs(stats: AttentionStats, output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{output_name}.pkl").open("wb") as handle:
        pickle.dump(stats.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and plot SADT/LLaVA attention statistics.")
    parser.add_argument("--repo-root", default=None, help="Path to the repository containing llava/ and chair.py.")
    parser.add_argument("--model-path", required=True, help="Local LLaVA checkpoint path.")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-dir", required=True, help="Directory containing evaluation images.")
    parser.add_argument("--labels-json", default=None, help="Optional JSON file with gt_words for each image.")
    parser.add_argument("--chair-cache", default="data/chair.pkl", help="Optional CHAIR evaluator pickle cache.")
    parser.add_argument("--coco-annotations", default=None, help="COCO annotation directory, used when cache is absent.")
    parser.add_argument("--output-dir", default="outputs/attention_vis")
    parser.add_argument("--output-name", default="vis_data")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--target-layers", default="19:26", help="Half-open range, e.g. 19:26.")
    parser.add_argument("--image-token-count", type=int, default=576)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stats = collect_statistics(args)
    save_outputs(stats, Path(args.output_dir), args.output_name)


if __name__ == "__main__":
    main()
