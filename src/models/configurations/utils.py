from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import tqdm


def handle_common(
    trials: dict[str, dict[str, Any]],
    keep_tag_if: Callable[[str], bool] | None = None,
    progress_bar: str | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    if keep_tag_if is not None:
        trials = {k: v for k, v in trials.items() if keep_tag_if(k)}
    trials = trials.items()
    if progress_bar is not None:

        def prog():
            bar = tqdm.tqdm(trials, desc=progress_bar, unit="kind")
            for desc, kwargs in bar:
                bar.set_postfix_str(desc)
                yield desc, kwargs

        return prog()
    return trials
