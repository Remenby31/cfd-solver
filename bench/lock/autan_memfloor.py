"""The host memory floor of autan-gpu-lock and autan-campaign.

On GB10 the GPU allocates from host memory, and a container's memory limit does not bound
device allocations: a run that exhausts host memory hangs the whole Spark. Both tools keep
MemAvailable (proc(5)) above a floor and kill what they run when it falls below.
"""

from __future__ import annotations

import re
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
DEFAULT = "24G"
_UNITS = {"G": 1 << 30, "M": 1 << 20}


def parse(text: str) -> int | None:
    """A size written `<whole number>G` or `<whole number>M`, in bytes; None if malformed."""
    match = re.fullmatch(r"(0|[1-9][0-9]{0,6})([GM])", text)
    return int(match[1]) * _UNITS[match[2]] if match else None


def available() -> int | None:
    """MemAvailable in bytes, or None if the kernel does not report it."""
    try:
        for line in MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def below(floor: int) -> bool:
    """Whether a nonzero floor is not met; an unknown MemAvailable does not meet it."""
    if not floor:
        return False
    free = available()
    return free is None or free < floor


def show(size: int | None) -> str:
    return "unknown" if size is None else f"{size / (1 << 30):.1f} GiB"
