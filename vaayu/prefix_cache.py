"""Cross-turn KV prefix caching (A1): reuse a session's KV cache across turns,
prefilling only the divergent tail. See
docs/superpowers/specs/2026-07-12-vaayu-prefix-caching-design.md."""
import threading


def longest_common_prefix(a: list, b: list) -> int:
    """Number of leading elements equal in both sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
