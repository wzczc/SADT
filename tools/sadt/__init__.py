"""SADT detection, classification, and mitigation utilities."""

from .core import (
    SADTConfig,
    SADTDetection,
    build_vcd_hallucination_guidance,
    detect_hallucinations,
    filter_words_by_image_attention,
    find_token_spans,
    generate_with_trace,
    mask_high_attention_regions_vcd,
)

__all__ = [
    "SADTConfig",
    "SADTDetection",
    "build_vcd_hallucination_guidance",
    "detect_hallucinations",
    "filter_words_by_image_attention",
    "find_token_spans",
    "generate_with_trace",
    "mask_high_attention_regions_vcd",
]
