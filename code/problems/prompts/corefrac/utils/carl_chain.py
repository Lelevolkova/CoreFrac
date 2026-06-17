"""Fixed CARL tool chain runner for SAM3 geological fracture segmentation."""

from __future__ import annotations

import os
from typing import Any

from problems.prompts.corefrac.utils.dataset import (
    CrackSample,
)
from problems.prompts.corefrac.utils.sam3_tool import (
    sam3_segment_cracks,
)


def _require_carl():
    try:
        from mmar_carl import ReasoningChain, ToolStepDescription
        from mmar_carl.models import Language, ReasoningContext, ToolStepConfig
        from mmar_carl.models.llm_client_base import LLMClientBase
    except ImportError as exc:
        raise ImportError(
            "mmar-carl is required for this problem. "
            "Install with `pip install 'gigaevo[chains]'` (Python >= 3.12)."
        ) from exc
    return (
        ReasoningChain,
        ToolStepDescription,
        Language,
        ReasoningContext,
        ToolStepConfig,
        LLMClientBase,
    )


def _build_tool_only_client():
    _, _, _, _, _, LLMClientBase = _require_carl()

    class ToolOnlyClient(LLMClientBase):
        async def get_response(self, prompt: str) -> str:
            return ""

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return ""

    return ToolOnlyClient()


_FIXED_CHAIN = None


def get_fixed_chain():
    global _FIXED_CHAIN
    if _FIXED_CHAIN is not None:
        return _FIXED_CHAIN

    ReasoningChain, ToolStepDescription, _, _, ToolStepConfig, _ = _require_carl()

    steps = [
        ToolStepDescription(
            number=1,
            title="Segment geological fracture mask with SAM3",
            config=ToolStepConfig(
                tool_name="sam3_segment_cracks",
                timeout=float(os.environ.get("SAM3_TOOL_TIMEOUT", "900")),
                input_mapping={
                    "image_path": "$metadata.image_path",
                    "text_prompt": "$metadata.sam_prompt",
                },
            ),
            dependencies=[],
        )
    ]
    _FIXED_CHAIN = ReasoningChain(steps=steps, max_workers=1)
    return _FIXED_CHAIN


def run_sam3_on_sample(sample: CrackSample, sam_prompt: str) -> dict[str, Any]:
    """Execute the fixed CARL tool chain for one CoreFrac sample."""
    _, _, Language, ReasoningContext, _, _ = _require_carl()

    chain = get_fixed_chain()
    context = ReasoningContext(
        outer_context=sample.sample_id,
        api=_build_tool_only_client(),
        language=Language.ENGLISH,
        metadata={
            "image_path": str(sample.image_path),
            "sam_prompt": sam_prompt,
        },
    )
    context.register_tool("sam3_segment_cracks", sam3_segment_cracks)

    result = chain.execute(context)
    if not result.success or not result.step_results:
        raise RuntimeError(
            f"CARL chain failed for sample {sample.sample_id}: "
            f"{result.error or 'unknown error'}"
        )

    step = result.step_results[0]
    if not step.success:
        raise RuntimeError(
            f"SAM3 tool step failed for sample {sample.sample_id}: "
            f"{step.error_message or 'unknown error'}"
        )

    data = step.result_data
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected SAM3 tool output type for sample {sample.sample_id}: "
            f"{type(data).__name__}"
        )
    return data
