from typing import TypeVar
from typing import Any

T = TypeVar("T")


def convertDictValuesToString(d: dict[T, Any]) -> dict[T, str]:
    return {k: str(v) for (k, v) in d.items()}
