"""SAM3 text prompt validation for evolved programs."""

from __future__ import annotations


def validate_sam3_prompt(
    prompt: str,
    *,
    max_length: int,
    forbidden_substrings: list[str],
) -> None:
    """Validate evolved SAM3 prompt constraints.

    Raises:
        ValueError: If prompt violates experiment constraints.
    """
    if not isinstance(prompt, str):
        raise ValueError(f"entrypoint() must return str, got {type(prompt).__name__}")

    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("SAM3 prompt must be non-empty")

    if len(cleaned) > max_length:
        raise ValueError(
            f"SAM3 prompt exceeds max length {max_length} (got {len(cleaned)})"
        )

    lowered = cleaned.lower()
    for forbidden in forbidden_substrings:
        if forbidden.lower() in lowered:
            raise ValueError(f"SAM3 prompt contains forbidden substring: {forbidden!r}")
