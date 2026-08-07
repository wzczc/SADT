"""Generate paper-style SADT visualizations.

Supported figures:
  - save-vis-data: collect dataset-level attention statistics for fig1/fig2.
  - save-pkl: cache model outputs/attentions/hidden states for later plotting.
  - fig1: visual-attention comparison for token categories.
  - fig2: cross-layer attention curves for token categories.
  - fig4: logit lens over image patches combined with visual attention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


try:
    import inflect
except ImportError:
    inflect = None

ENGINE = inflect.engine() if inflect is not None else None


try:
    import torch
except ImportError:
    torch = None


def require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for fig4 model inference.")
    return torch


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for fig4 visual overlays.") from exc
    return cv2


def parse_layer_range(value: str) -> range:
    start_text, end_text = value.split(":", maxsplit=1)
    return range(int(start_text), int(end_text))


def sanitize_filename(value: str) -> str:
    value = value.replace("</s>", "EOS").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "token"


def png_output_path(path: Path) -> Path:
    """Keep paper-figure outputs in PNG format."""
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")


def infer_num_layers(model_path: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return fallback
    lowered = model_path.lower()
    if "13b" in lowered:
        return 40
    if "7b" in lowered:
        return 32
    raise ValueError("Please pass --num-layers for this model.")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def cpuify(value):
    """Move tensors in a nested structure to CPU before pickling."""

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpuify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpuify(item) for item in value)
    if isinstance(value, list):
        return [cpuify(item) for item in value]
    return value


def image_files_from_args(image_file: str | None, image_dir: str | None, limit: int | None) -> list[Path]:
    if image_file:
        return [Path(image_file)]
    if not image_dir:
        raise ValueError("Pass either --image-file or --image-dir.")
    image_paths = sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    return image_paths[:limit] if limit is not None else image_paths


def load_visualize_attention_module():
    module_path = Path(__file__).with_name("visualize_attention.py")
    spec = importlib.util.spec_from_file_location("sadt_visualize_attention", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_visualization_data(args: argparse.Namespace) -> None:
    """Create the dataset-level vis_data pkl used by Figure 1 and Figure 2."""

    visualize_attention = load_visualize_attention_module()
    stats = visualize_attention.collect_statistics(args)
    visualize_attention.save_outputs(stats, Path(args.output_dir), args.output_name)


def find_token_sequence_indices(target_ids: Sequence[int], generated_ids: Sequence[int]) -> list[int]:
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
    generated_list = generated_ids[1:].tolist()
    word_indices = []
    for word in words:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        indices = find_token_sequence_indices(token_ids, generated_list)
        if not indices and ENGINE is not None:
            plural = ENGINE.plural(word)
            token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"]
            indices = find_token_sequence_indices(token_ids, generated_list)
        word_indices.append(indices)
    return word_indices


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def plot_fig1_visual_attention_comparison(stats: dict, save_path: Path) -> None:
    """Figure 1: compare visual attention for real/hallucinated/non-object tokens."""

    save_path = png_output_path(save_path)

    groups = [
        ("Real", stats.get("all_real_attentions", [])),
        ("Hallucinated", stats.get("all_hallu_attentions", [])),
        ("Non-object", stats.get("all_non_attentions", [])),
    ]
    groups = [(name, values) for name, values in groups if values]
    if not groups:
        raise ValueError("No visual-attention values found in stats pkl.")

    labels = [name for name, _ in groups]
    means = [float(np.mean(values)) for _, values in groups]
    stds = [float(np.std(values)) for _, values in groups]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    colors = ["#4C78A8", "#E45756", "#54A24B"][: len(groups)]
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.82, width=0.58)
    ax.set_ylabel("Visual Attention Ratio")
    ax.set_xlabel("Token Type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def average_curve(stats: dict, key: str, num_layers: int) -> np.ndarray:
    values = stats.get(key, [])
    if not values:
        raise ValueError(f"Missing or empty key in stats pkl: {key}")
    curve = np.asarray(values, dtype=np.float32).mean(axis=0)
    if len(curve) != num_layers:
        raise ValueError(f"{key} has {len(curve)} layers, but --num-layers is {num_layers}.")
    return curve


def fig2_xticks(num_layers: int) -> list[int]:
    if num_layers == 32:
        return [1, 4, 8, 12, 16, 20, 24, 28, 32]
    if num_layers == 40:
        return [1, 5, 10, 15, 20, 25, 30, 35, 40]
    if num_layers == 28:
        return [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    return list(np.linspace(1, num_layers, 6, dtype=int))


def plot_fig2_cross_layer_attention(
    stats: dict,
    save_path: Path,
    num_layers: int,
    stage_start: int | None = None,
    stage_end: int | None = None,
) -> None:
    """Figure 2: System/Image/Text attention across layers.

    This follows the original paper_vis/plot_test*.py style: subplot (a) shows
    object tokens, subplot (b) shows non-object tokens, and subplot (a)
    highlights the image-attention stage.
    """

    save_path = png_output_path(save_path)
    layers = np.arange(1, num_layers + 1)
    stage_start = stage_start or (22 if num_layers == 40 else 20)
    stage_end = stage_end or (29 if num_layers == 40 else 27)

    object_sys = average_curve(stats, "all_att_sys", num_layers)
    object_img = average_curve(stats, "all_att_img", num_layers)
    object_text = average_curve(stats, "all_att_tex", num_layers)
    non_sys = average_curve(stats, "non_att_sys", num_layers)
    non_img = average_curve(stats, "non_att_img", num_layers)
    non_text = average_curve(stats, "non_att_tex", num_layers)

    curves = [object_sys, object_img, object_text, non_sys, non_img, non_text]
    y_min = min(float(curve.min()) for curve in curves)
    y_max = max(float(curve.max()) for curve in curves)
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
    y_lim = (y_min - margin, y_max + margin)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "lines.markersize": 3.5,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.7,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        }
    ):
        sys_color = "#C49365"
        colors = plt.get_cmap("Set1").colors
        img_color = colors[1]
        text_color = colors[2]

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(7.5, 7.5))

        def draw_panel(ax, sys_curve, img_curve, text_curve, tag: str, highlight: bool) -> None:
            ax.plot(layers, sys_curve, marker="o", linestyle="-", label="System", color=sys_color)
            ax.plot(layers, img_curve, marker="s", linestyle="--", label="Image", color=img_color)
            ax.plot(layers, text_curve, marker="^", linestyle=":", label="Text", color=text_color)
            if highlight:
                stage_mask = (layers >= stage_start) & (layers <= stage_end)
                fill_mask = stage_mask & (img_curve > text_curve)
                ax.axvspan(stage_start - 0.5, stage_end + 0.5, color=img_color, alpha=0.18, zorder=0)
                ax.fill_between(
                    layers,
                    img_curve,
                    text_curve,
                    where=fill_mask,
                    interpolate=True,
                    color=img_color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=1,
                    label="Image Attention Stage",
                )
                if fill_mask.any():
                    ax.plot(layers[fill_mask], img_curve[fill_mask], color=img_color, linewidth=2.2, zorder=3)
                    ax.scatter(
                        layers[fill_mask],
                        img_curve[fill_mask],
                        color=img_color,
                        edgecolor="black",
                        s=26,
                        zorder=4,
                    )

            ax.set_ylabel("Attention Ratio", fontweight="bold")
            ax.set_xlim(0, num_layers + 1)
            ax.set_ylim(*y_lim)
            ax.set_xticks(fig2_xticks(num_layers))
            ax.minorticks_on()
            ax.text(0.01, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontweight="bold")

        draw_panel(axes[0], object_sys, object_img, object_text, "(a)", highlight=True)
        draw_panel(axes[1], non_sys, non_img, non_text, "(b)", highlight=False)
        axes[1].set_xlabel("Layer", fontweight="bold")

        from matplotlib.font_manager import FontProperties

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=True,
            prop=FontProperties(weight="bold"),
            fancybox=False,
            framealpha=0.9,
            edgecolor="black",
            bbox_to_anchor=(0.54, 0.95),
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def visual_attention_grid(
    attentions,
    token_idx: int,
    img_idx: int,
    layers: range,
    image_token_count: int,
    grid_size: int,
) -> np.ndarray:
    torch_mod = require_torch()
    attention_layers = torch_mod.stack(attentions[token_idx])
    attention_img = (
        attention_layers[list(layers), 0, :, -1, img_idx : img_idx + image_token_count]
        .detach()
        .float()
        .cpu()
        .numpy()
        .mean(axis=1)
        .mean(axis=0)
    )
    return attention_img.reshape(grid_size, grid_size)


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    return (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)


def overlay_attention(image: Image.Image, attention_grid: np.ndarray, alpha: float = 0.55) -> Image.Image:
    cv2 = require_cv2()
    image = image.convert("RGB").resize((336, 336))
    img_np = np.array(image)
    att_resized = cv2.resize(normalize_grid(attention_grid).astype(np.float32), (336, 336))
    heatmap = cv2.applyColorMap((att_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def logit_lens_for_image_patches(
    hidden_states,
    model,
    tokenizer,
    img_idx: int,
    layer: int,
    image_token_count: int,
    topk: int,
) -> tuple[list[list[str]], list[list[float]]]:
    torch_mod = require_torch()
    input_hidden_states = hidden_states[0]
    visual_hidden = input_hidden_states[layer][:, img_idx : img_idx + image_token_count, :]
    logits = model.lm_head(visual_hidden)
    probs = torch_mod.softmax(logits, dim=-1)
    top_probs, top_ids = torch_mod.topk(probs, k=topk, dim=-1)

    all_tokens: list[list[str]] = [[] for _ in range(topk)]
    all_probs: list[list[float]] = [[] for _ in range(topk)]
    for patch_idx in range(image_token_count):
        for rank in range(topk):
            token_id = top_ids[0, patch_idx, rank].item()
            token = tokenizer.decode([token_id], skip_special_tokens=False).strip()
            all_tokens[rank].append(token)
            all_probs[rank].append(float(top_probs[0, patch_idx, rank].item()))
    return all_tokens, all_probs


def draw_logit_lens_attention_panel(
    image: Image.Image,
    attention_grid: np.ndarray,
    lens_tokens: Sequence[str],
    lens_probs: Sequence[float],
    top_patch_indices: Sequence[int],
    save_path: Path,
    title: str,
    grid_size: int = 24,
) -> None:
    overlay = overlay_attention(image, attention_grid)
    draw = ImageDraw.Draw(overlay)
    patch_size = overlay.size[0] / grid_size

    for patch_idx in top_patch_indices:
        row = int(patch_idx) // grid_size
        col = int(patch_idx) % grid_size
        x1 = col * patch_size
        y1 = row * patch_size
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
        text = f"{lens_tokens[patch_idx]}\n{lens_probs[patch_idx] * 100:.1f}%"
        draw.rectangle([x1, y1, min(x1 + 86, overlay.size[0]), min(y1 + 31, overlay.size[1])], fill=(255, 255, 255))
        draw.text((x1 + 2, y1 + 2), text, fill=(0, 0, 0))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_font(candidates: Sequence[str], size: int):
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_fig4_layer_token_maps(
    image: Image.Image,
    words: Sequence[str],
    attentions,
    render_layers: range,
    stage_layers: range,
    img_idx: int,
    word_token_indices: Sequence[Sequence[int]],
    hidden_states,
    model,
    tokenizer,
    output_dir: Path,
    image_name: str,
    image_token_count: int,
    grid_size: int,
    topk: int,
    boost: float,
    single_example: bool,
) -> dict[str, dict]:
    """Render Fig. 4 maps following paper_vis/attn_map_vis.py.

    For every target word and every selected layer, this saves one enlarged
    visual-attention overlay. The top attention patches in that layer are
    annotated with logit-lens tokens from the corresponding visual hidden
    states.
    """

    torch_mod = require_torch()
    cv2 = require_cv2()
    img_size = (336, 336)
    extended_size = (672, 672)
    original_img = image.convert("RGB").resize(img_size)
    extended_img = original_img.resize(extended_size)
    img_bgr_extended = cv2.cvtColor(np.array(extended_img), cv2.COLOR_RGB2BGR)

    # LLaVA generation returns hidden_states as (generation_step, layer, ...).
    # Fig. 4 uses visual patch states from the input forward pass.
    input_hidden_states = hidden_states[0]
    layer_to_grid_tokens = {}
    with torch_mod.no_grad():
        for layer_idx in render_layers:
            hidden_layer_idx = layer_idx + 1
            visual_hidden = input_hidden_states[hidden_layer_idx][:, img_idx : img_idx + image_token_count, :].to(
                model.device
            )
            visual_logits = model.lm_head(visual_hidden)
            visual_token_ids = torch_mod.argmax(visual_logits, dim=-1).squeeze(0).detach().cpu().numpy()
            visual_tokens = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in visual_token_ids]
            layer_to_grid_tokens[layer_idx] = np.array(visual_tokens).reshape(grid_size, grid_size)

    font = load_font(["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"], size=20)
    token_size = extended_size[0] // grid_size
    save_root = output_dir / "fig4_attention_map" / image_name
    summary: dict[str, dict] = {}

    for word, indices in zip(words, word_token_indices):
        if not indices:
            summary[word] = {"found": False}
            continue

        token_idx = indices[0]
        attention_layers = torch_mod.stack(attentions[token_idx])
        attention_img = (
            attention_layers[:, 0, :, -1, img_idx : img_idx + image_token_count]
            .detach()
            .float()
            .cpu()
            .numpy()
            .mean(axis=1)
        )
        global_min = attention_img.min()
        global_max = attention_img.max()

        word_dir = save_root / f"{sanitize_filename(word)}_layers_with_token"
        if not single_example:
            word_dir.mkdir(parents=True, exist_ok=True)
        saved_layers = []

        for layer_idx in render_layers:
            att_flat = attention_img[layer_idx]
            att_norm_orig = (att_flat - global_min) / (global_max - global_min + 1e-8)
            att_norm_orig = np.clip(att_norm_orig * boost, 0.0, 1.0)
            threshold = np.percentile(att_norm_orig, 95)
            att_norm = att_norm_orig.copy()
            att_norm[att_norm < threshold] = 0
            att_grid = att_norm.reshape(grid_size, grid_size).astype(np.float32)

            att_resized = cv2.resize(att_grid, extended_size, interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap((att_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay_bgr = cv2.addWeighted(heatmap_color, 0.45, img_bgr_extended, 0.55, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            overlay = Image.fromarray(overlay_rgb)
            draw = ImageDraw.Draw(overlay)

            k = min(topk, att_flat.size)
            top_idx = np.argpartition(att_flat, -k)[-k:]
            top_idx = top_idx[np.argsort(-att_flat[top_idx])]
            current_grid_tokens = layer_to_grid_tokens[layer_idx]
            top_patch_records = []
            for flat_idx in top_idx:
                row, col = divmod(int(flat_idx), grid_size)
                token_text = current_grid_tokens[row][col]
                x = col * token_size + token_size // 2
                y = row * token_size + token_size // 2
                draw.text((x, y), token_text, fill=(255, 255, 255), anchor="mm", font=font)
                top_patch_records.append(
                    {
                        "patch_index": int(flat_idx),
                        "logit_lens_token": token_text,
                        "attention": float(att_flat[int(flat_idx)]),
                    }
                )

            if single_example:
                save_root.mkdir(parents=True, exist_ok=True)
                save_path = save_root / f"{sanitize_filename(word)}_layer_{layer_idx}.png"
            else:
                save_path = word_dir / f"layer_{layer_idx}.png"
            overlay.save(save_path, dpi=(300, 300))
            saved_layers.append(
                {
                    "layer": int(layer_idx),
                    "figure": str(save_path),
                    "top_patches": top_patch_records,
                }
            )

        summary[word] = {
            "found": True,
            "generated_token_index": int(token_idx),
            "layers": saved_layers,
        }

    return summary


def load_llava(model_path: str, model_base: str | None, repo_root: str | None):
    require_torch()
    if repo_root:
        sys.path.insert(0, str(Path(repo_root).resolve()))
    from llava.eval.run_llava import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    disable_torch_init()
    return load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
    )


def run_llava_image(
    image_file: Path,
    prompt: str,
    model_path: str,
    model_base: str | None,
    tokenizer,
    model,
    image_processor,
    max_new_tokens: int,
):
    torch_mod = require_torch()
    from llava.eval.run_llava import eval_model
    from llava.mm_utils import get_model_name_from_path

    image = Image.open(image_file).convert("RGB")
    image_sizes = [image.size]
    square_image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(square_image, return_tensors="pt")["pixel_values"][0].to(
        model.device,
        dtype=torch_mod.float16,
    )
    args = type(
        "Args",
        (),
        {
            "model_base": model_base,
            "model_path": model_path,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": str(image_file),
            "image_sizes": image_sizes,
            "images_tensor": [image_tensor],
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        },
    )()
    output, input_ids = eval_model(args)
    generated_ids = output["sequences"][0]
    img_idx = torch_mod.nonzero(input_ids[0] == -200).squeeze().item()
    system_prompt_ids = input_ids[0][:img_idx]
    query_prompt_ids = input_ids[0][img_idx + 1 :]
    return {
        "output": output,
        "input_ids": input_ids,
        "generated_ids": generated_ids,
        "generate_tokens": tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
        "caption": tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0],
        "img_idx": img_idx,
        "system_prompt_ids": system_prompt_ids,
        "query_prompt_ids": query_prompt_ids,
        "system_prompt": tokenizer.batch_decode(system_prompt_ids, skip_special_tokens=True),
        "query_prompt": tokenizer.batch_decode(query_prompt_ids, skip_special_tokens=True),
        "image": image,
        "image_path": image_file,
    }


def load_cached_run(cache_pkl: Path) -> dict:
    """Load a single-image cache produced by ``save-pkl``."""

    record = load_pickle(cache_pkl)
    image_path = Path(record["img_path"])
    return {
        "output": {
            "attentions": record["attentions"],
            "hidden_states": record["hidden_states"],
            "sequences": record["sequences"],
        },
        "input_ids": record["input_ids"],
        "generated_ids": record["generated_ids"],
        "generate_tokens": record["generate_text"],
        "caption": record["generate_text_clean"],
        "img_idx": record["img_idx"],
        "system_prompt_ids": record["system_prompt_ids"],
        "query_prompt_ids": record["query_prompt_ids"],
        "system_prompt": record["system_prompt"],
        "query_prompt": record["query_prompt"],
        "image": Image.open(image_path).convert("RGB"),
        "image_path": image_path,
    }


def build_attention_cache_record(run: dict, image_path: Path, prompt: str) -> dict:
    """Build a pkl record compatible with the original save_attn.py fields."""

    output = run["output"]
    record = {
        "img_path": str(image_path),
        "prompt": prompt,
        "input_ids": run["input_ids"],
        "generated_ids": run["generated_ids"],
        "generate_text": run["generate_tokens"],
        "generate_text_clean": run["caption"],
        "img_idx": run["img_idx"],
        "system_prompt_ids": run["system_prompt_ids"],
        "query_prompt_ids": run["query_prompt_ids"],
        "system_prompt": run["system_prompt"],
        "query_prompt": run["query_prompt"],
        "attentions": output["attentions"],
        "hidden_states": output["hidden_states"],
        "sequences": output["sequences"],
    }
    return cpuify(record)


def save_attention_cache(args: argparse.Namespace) -> None:
    tokenizer, model, image_processor, _ = load_llava(args.model_path, args.model_base, args.repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for image_path in image_files_from_args(args.image_file, args.image_dir, args.limit):
        run = run_llava_image(
            image_path,
            args.prompt,
            args.model_path,
            args.model_base,
            tokenizer,
            model,
            image_processor,
            args.max_new_tokens,
        )
        record = build_attention_cache_record(run, image_path, args.prompt)
        save_path = output_dir / f"{image_path.stem}.pkl"
        with save_path.open("wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        summary[image_path.name] = {
            "pkl": str(save_path),
            "caption": run["caption"],
            "num_generated_tokens": len(run["generate_tokens"]),
        }
        print(f"saved {save_path}")

    with (output_dir / "captions.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def generate_fig4(args: argparse.Namespace) -> None:
    tokenizer, model, image_processor, _ = load_llava(args.model_path, args.model_base, args.repo_root)
    if args.cache_pkl:
        run = load_cached_run(Path(args.cache_pkl))
    else:
        if not args.image_file:
            raise ValueError("Pass --image-file, or pass --cache-pkl from save-pkl.")
        run = run_llava_image(
            Path(args.image_file),
            args.prompt,
            args.model_path,
            args.model_base,
            tokenizer,
            model,
            image_processor,
            args.max_new_tokens,
        )

    words = args.words
    word_token_indices = locate_words_in_generation(tokenizer, words, run["generated_ids"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_layers = parse_layer_range(args.target_layers)
    if args.example_layer is not None:
        render_layers = range(args.example_layer, args.example_layer + 1)
    elif not args.all_layers:
        render_layers = range(23, 24)
    else:
        render_layers = stage_layers
    image_name = Path(run.get("image_path", args.image_file)).stem
    render_fig4_layer_token_maps(
        image=run["image"],
        words=words,
        attentions=run["output"]["attentions"],
        render_layers=render_layers,
        stage_layers=stage_layers,
        img_idx=run["img_idx"],
        word_token_indices=word_token_indices,
        hidden_states=run["output"]["hidden_states"],
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        image_name=image_name,
        image_token_count=args.image_token_count,
        grid_size=args.grid_size,
        topk=args.top_patches,
        boost=args.attention_boost,
        single_example=not args.all_layers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SADT paper visualizations.")
    subparsers = parser.add_subparsers(dest="figure", required=True)

    save_vis_data = subparsers.add_parser(
        "save-vis-data",
        help="Collect dataset-level statistics and save vis_data pkl for fig1/fig2.",
    )
    save_vis_data.add_argument("--repo-root", default=None, help="Path containing llava/ and chair.py.")
    save_vis_data.add_argument("--model-path", required=True)
    save_vis_data.add_argument("--model-base", default=None)
    save_vis_data.add_argument("--image-dir", default="data/test_imgs")
    save_vis_data.add_argument("--labels-json", default="data/metadata/hallucination_results_val500.json")
    save_vis_data.add_argument("--chair-cache", default="data/chair.pkl")
    save_vis_data.add_argument("--coco-annotations", default=None)
    save_vis_data.add_argument("--output-dir", default="outputs/llava7b_attention")
    save_vis_data.add_argument("--output-name", default="vis_data_7b")
    save_vis_data.add_argument("--prompt", default="Describe this image.")
    save_vis_data.add_argument("--conv-mode", default=None)
    save_vis_data.add_argument("--temperature", type=float, default=0.0)
    save_vis_data.add_argument("--top-p", type=float, default=None)
    save_vis_data.add_argument("--num-beams", type=int, default=1)
    save_vis_data.add_argument("--max-new-tokens", type=int, default=256)
    save_vis_data.add_argument("--num-layers", type=int, default=None)
    save_vis_data.add_argument("--target-layers", default="19:26")
    save_vis_data.add_argument("--image-token-count", type=int, default=576)
    save_vis_data.add_argument("--limit", type=int, default=None)

    save_pkl = subparsers.add_parser("save-pkl", help="Cache attentions and hidden states as pkl files.")
    save_pkl.add_argument("--repo-root", default=None)
    save_pkl.add_argument("--model-path", required=True)
    save_pkl.add_argument("--model-base", default=None)
    save_pkl.add_argument("--image-file", default=None)
    save_pkl.add_argument("--image-dir", default="data/test_imgs")
    save_pkl.add_argument("--prompt", default="Describe this image.")
    save_pkl.add_argument("--max-new-tokens", type=int, default=256)
    save_pkl.add_argument("--output-dir", default="outputs/attention_cache")
    save_pkl.add_argument("--limit", type=int, default=None)

    fig1 = subparsers.add_parser("fig1", help="Visual-attention comparison.")
    fig1.add_argument("--stats-pkl", default="outputs/llava7b_attention/vis_data_7b.pkl")
    fig1.add_argument("--output", default="outputs/paper_figures/fig1_visual_attention.png")

    fig2 = subparsers.add_parser("fig2", help="Cross-layer attention curves.")
    fig2.add_argument("--stats-pkl", default="outputs/llava7b_attention/vis_data_7b.pkl")
    fig2.add_argument("--output", default="outputs/paper_figures/fig2_cross_layer_attention.png")
    fig2.add_argument("--num-layers", type=int, default=32)
    fig2.add_argument("--stage-start", type=int, default=20)
    fig2.add_argument("--stage-end", type=int, default=27)

    fig4 = subparsers.add_parser("fig4")
    fig4.add_argument("--repo-root", default=None)
    fig4.add_argument("--model-path", required=True)
    fig4.add_argument("--model-base", default=None)
    fig4.add_argument("--image-file", default="data/test_imgs/COCO_val2014_000000007795.jpg")
    fig4.add_argument("--cache-pkl", default=None, help="Optional single-image pkl produced by save-pkl.")
    fig4.add_argument("--prompt", default="Describe this image.")
    fig4.add_argument("--words", nargs="+", required=True, help="Target generated words, usually hallucinated words.")
    fig4.add_argument("--target-layers", default="19:26")
    fig4.add_argument("--image-token-count", type=int, default=576)
    fig4.add_argument("--grid-size", type=int, default=24)
    fig4.add_argument("--top-patches", type=int, default=3)
    fig4.add_argument("--max-new-tokens", type=int, default=256)
    fig4.add_argument("--output-dir", default="outputs/paper_figures")
    fig4.add_argument("--attention-boost", type=float, default=1.5)
    fig4.add_argument("--example-layer", type=int, default=None, help="Layer used for the paper-style single example.")
    fig4.add_argument("--all-layers", action="store_true", help="Render every layer in --target-layers for debugging.")
    fig4.set_defaults(target_layers="19:26", top_patches=3)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.figure == "save-vis-data":
        save_visualization_data(args)
    elif args.figure == "save-pkl":
        save_attention_cache(args)
    elif args.figure == "fig1":
        plot_fig1_visual_attention_comparison(load_pickle(Path(args.stats_pkl)), Path(args.output))
    elif args.figure == "fig2":
        plot_fig2_cross_layer_attention(
            load_pickle(Path(args.stats_pkl)),
            Path(args.output),
            args.num_layers,
            args.stage_start,
            args.stage_end,
        )
    elif args.figure == "fig4":
        generate_fig4(args)


if __name__ == "__main__":
    main()
