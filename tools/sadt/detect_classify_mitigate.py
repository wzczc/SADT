"""Run SADT detection, classification, and mitigation in one process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

from tools.sadt.core import (
    SADTConfig,
    build_semantic_forms,
    build_vcd_hallucination_guidance,
    detect_hallucinations,
    filter_words_by_image_attention,
    generate_with_trace,
)
from tools.sadt.pipeline_utils import (
    SEMANTIC_MATCHER_CHOICES,
    chair_output_record,
    chair_summary,
    chair_words,
    choose_words,
    classification_summary,
    collect_images,
    detection_summary,
    label_hallucination_words,
    load_chair_evaluator,
    load_labels,
    mask_only_words,
    parse_layers,
    parse_words,
    prepare_vcd_image_tensor,
    save_json_output,
    set_detection_types_from_words,
)
from tools.sadt.vcd_sample import temporary_vcd_sampling


def run_one_image(
    *,
    image_file: str,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    model_name: str,
    prompt: str,
    config: SADTConfig,
    explicit_words: Optional[Sequence[str]],
    label_item: Optional[Dict[str, Any]],
    chair_evaluator: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: Optional[float],
    vcd_hidden_layer_index: int,
) -> Dict[str, Any]:
    image_name = Path(image_file).name

    original_output, input_ids, original_response = generate_with_trace(
        model,
        tokenizer,
        image_processor,
        model_name,
        image_file,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    original_image_sizes = [Image.open(image_file).size]
    original_chair = chair_words(chair_evaluator, image_name, original_response)
    word_info = choose_words(
        explicit_words=explicit_words,
        label_item=label_item,
        chair_item=original_chair,
    )
    gt_words = word_info["gt_words"]
    original_record = chair_output_record(
        evaluator=chair_evaluator,
        image_name=image_name,
        response=original_response,
        gt_words=gt_words,
        excluded_words=config.excluded_words,
    )

    attention_filter = filter_words_by_image_attention(
        words=word_info["generated_words"],
        output=original_output,
        input_ids=input_ids,
        tokenizer=tokenizer,
        config=config,
    )
    object_words = [
        word for word in attention_filter["object_words"] if word not in config.excluded_words
    ]
    detections = detect_hallucinations(
        words=object_words,
        output=original_output,
        input_ids=input_ids,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    detected_hallucinations = [
        item.word for item in detections if item.is_hallucination and item.word not in config.excluded_words
    ]
    hallucination_info = label_hallucination_words(
        label_item=label_item,
        excluded_words=config.excluded_words,
    )

    classification_mask = mask_only_words(
        image_file=image_file,
        detections=detections,
        words=detected_hallucinations,
        config=config,
        image_mean=getattr(image_processor, "image_mean", None),
    )

    harm_record = None
    type1_words = []
    type2_words = []
    if classification_mask:
        classification_tensor, _ = prepare_vcd_image_tensor(
            classification_mask,
            image_processor,
            model,
            image_sizes=original_image_sizes,
        )
        _, _, harm_response = generate_with_trace(
            model,
            tokenizer,
            image_processor,
            model_name,
            image_file,
            prompt,
            image_tensor=classification_tensor,
            image_sizes=original_image_sizes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        harm_record = chair_output_record(
            evaluator=chair_evaluator,
            image_name=image_name,
            response=harm_response,
            gt_words=gt_words,
            excluded_words=config.excluded_words,
        )
        harm_hallu = set(harm_record["hallucination_words"])
        type2_words = [word for word in detected_hallucinations if word in harm_hallu]
        type1_words = [word for word in detected_hallucinations if word not in harm_hallu]
    set_detection_types_from_words(detections, type1_words=type1_words, type2_words=type2_words)

    mitigation_method = "none"
    mitigation_image = None
    mitigation_response = original_response
    warnings = []

    if type1_words:
        mitigation_image = mask_only_words(
            image_file=image_file,
            detections=detections,
            words=type1_words,
            config=config,
            image_mean=getattr(image_processor, "image_mean", None),
        )
        if mitigation_image:
            mitigation_method = "harm"
            mitigation_tensor, _ = prepare_vcd_image_tensor(
                mitigation_image,
                image_processor,
                model,
                image_sizes=original_image_sizes,
            )
            _, _, mitigation_response = generate_with_trace(
                model,
                tokenizer,
                image_processor,
                model_name,
                image_file,
                prompt,
                image_tensor=mitigation_tensor,
                image_sizes=original_image_sizes,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    if type2_words:
        hallu_indices, hallu_logits = build_vcd_hallucination_guidance(
            detections=detections,
            output=original_output,
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            config=config,
            hidden_layer_index=vcd_hidden_layer_index,
        )
        if hallu_indices and hallu_logits:
            mitigation_source_image = mitigation_image if mitigation_image is not None else image_file
            mitigation_source_tensor, _ = prepare_vcd_image_tensor(
                mitigation_source_image,
                image_processor,
                model,
                image_sizes=original_image_sizes,
            )
            with temporary_vcd_sampling():
                _, _, mitigation_response = generate_with_trace(
                    model,
                    tokenizer,
                    image_processor,
                    model_name,
                    image_file,
                    prompt,
                    image_tensor=mitigation_source_tensor,
                    image_sizes=original_image_sizes,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    hallu_indices=hallu_indices,
                    hallu_logits=hallu_logits,
                    cd_alpha=config.veed_alpha,
                )
            mitigation_method = "harm+veed" if mitigation_image is not None else "veed"
        else:
            warnings.append("no_veed_guidance_built_for_type2_words")

    mitigated_record = chair_output_record(
        evaluator=chair_evaluator,
        image_name=image_name,
        response=mitigation_response,
        gt_words=gt_words,
        excluded_words=config.excluded_words,
    )

    item = {
        "img_id": image_name,
        "image_file": image_file,
        "prompt": prompt,
        "gt_words_source": word_info["gt_words_source"],
        "gt_words": gt_words,
        "original_output": original_record,
        "detection": {
            "candidate_words_source": word_info["generated_words_source"],
            "image_attention_threshold": config.image_attention_threshold,
            "image_attention_scores": attention_filter["attention_scores"],
            "object_words": object_words,
            "hallucination_words_source": hallucination_info["hallucination_words_source"],
            "hallucination_words": hallucination_info["hallucination_words"],
            "detected_hallucinations": detected_hallucinations,
            "semantic_sets": {word: build_semantic_forms(word, config) for word in object_words},
            "detections": [asdict(item) for item in detections],
        },
        "classification": {
            "method": "harm_chair_persistence",
            "harm_output": harm_record,
            "type1_hallucinations": type1_words,
            "type2_hallucinations": type2_words,
        },
        "mitigated_output": {
            "method": mitigation_method,
            **mitigated_record,
        },
    }
    if warnings:
        item["mitigation_warnings"] = warnings
    return item


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-file", default=None)
    parser.add_argument("--image-dir", default="data/test_imgs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--output-json", default="outputs/sadt_full_pipeline/results.json")
    parser.add_argument(
        "--candidate-objects",
        default=None,
        help="Manual debugging override for generated words. Original experiments use CHAIR.",
    )
    parser.add_argument(
        "--labels-json",
        default="data/metadata/hallucination_results_val500.json",
        help="JSON providing gt_words and hallucination_words for detection metrics.",
    )
    parser.add_argument("--chair-cache", default="data/chair.pkl")
    parser.add_argument("--coco-annotations", default=None)
    parser.add_argument("--image-attention-layers", default="20:27")
    parser.add_argument("--image-attention-threshold", type=float, default=0.0)
    parser.add_argument("--topk-patches", type=int, default=4)
    parser.add_argument("--patches-to-mask-per-word", type=int, default=2)
    parser.add_argument("--model-image-size", type=int, default=336)
    parser.add_argument("--image-token-count", type=int, default=576)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--semantic-matcher", choices=SEMANTIC_MATCHER_CHOICES, default="coco")
    parser.add_argument("--wordnet-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--llm-judge-api-base", default=None)
    parser.add_argument("--llm-judge-model", default=None)
    parser.add_argument("--llm-judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-judge-timeout", type=float, default=20.0)
    parser.add_argument("--veed-alpha", type=float, default=0.0)
    parser.add_argument(
        "--vcd-hidden-layer-index",
        type=int,
        default=-8,
        help="Hidden-state layer index used to build VCD visual logits. The original VCD code uses -8.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    return parser


def compact_result(item: Dict[str, Any]) -> Dict[str, Any]:
    original_output = item.get("original_output") or {}
    detection = item.get("detection") or {}
    classification = item.get("classification") or {}
    return {
        "image_file": item.get("image_file"),
        "prompt": item.get("prompt"),
        "original_output": original_output.get("text"),
        "generated_words": original_output.get("generated_words") or [],
        "gt_words": item.get("gt_words") or [],
        "hallucination_words": detection.get("hallucination_words") or [],
        "detected_hallucination_words": detection.get("detected_hallucinations") or [],
        "type1_words": classification.get("type1_hallucinations") or [],
        "type2_words": classification.get("type2_hallucinations") or [],
    }


def main():
    args = build_parser().parse_args()
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
    )

    config = SADTConfig(
        image_attention_layers=parse_layers(args.image_attention_layers),
        image_attention_threshold=args.image_attention_threshold,
        topk_patches=args.topk_patches,
        patches_to_mask_per_word=args.patches_to_mask_per_word,
        model_image_size=args.model_image_size,
        image_token_count=args.image_token_count,
        grid_size=args.grid_size,
        veed_alpha=args.veed_alpha,
        semantic_matcher=args.semantic_matcher,
        wordnet_similarity_threshold=args.wordnet_similarity_threshold,
        llm_judge_api_base=args.llm_judge_api_base,
        llm_judge_model=args.llm_judge_model,
        llm_judge_api_key_env=args.llm_judge_api_key_env,
        llm_judge_timeout=args.llm_judge_timeout,
    )

    labels = load_labels(args.labels_json)
    chair_evaluator = load_chair_evaluator(args.chair_cache, args.coco_annotations)
    if chair_evaluator is None:
        raise ValueError("This pipeline requires CHAIR. Pass --chair-cache or --coco-annotations.")

    images = collect_images(args.image_file, args.image_dir, args.limit)
    explicit_words = parse_words(args.candidate_objects)
    output_json = args.output_json

    results = []
    for image_file in tqdm(images):
        image_name = Path(image_file).name
        item = run_one_image(
            image_file=image_file,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_name=model_name,
            prompt=args.prompt,
            config=config,
            explicit_words=explicit_words,
            label_item=labels.get(image_name),
            chair_evaluator=chair_evaluator,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            vcd_hidden_layer_index=args.vcd_hidden_layer_index,
        )
        results.append(item)
        print(
            f"{image_name}: detected={item['detection']['detected_hallucinations']} "
            f"type1={item['classification']['type1_hallucinations']} "
            f"type2={item['classification']['type2_hallucinations']} "
            f"mitigation={item['mitigated_output']['method']}"
        )

    output = {
        "results": [compact_result(item) for item in results],
        "metrics": {
            "detection": detection_summary(results),
            "original_chair": chair_summary(results, "original_output"),
            "classification": classification_summary(results),
            "mitigated_chair": chair_summary(results, "mitigated_output"),
        },
    }
    save_json_output(output_json, output)
    print(f"Saved: {output_json}")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
