from __future__ import annotations

import math


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    if not all(math.isfinite(value) for value in left + right):
        return None

    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0 or right_magnitude == 0:
        return None

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    return dot_product / (left_magnitude * right_magnitude)
