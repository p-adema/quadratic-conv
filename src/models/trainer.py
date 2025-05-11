from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import NamedTuple, Self

import numpy as np
import polars as pl
import torch
from sklearn.metrics import classification_report
from torch import nn
from tqdm.auto import tqdm, trange

from ..load_data import Dataset
from .utils import reports_to_df, split_seed

warnings.simplefilter("ignore", FutureWarning, 1725)


class FitManyResult(NamedTuple):
    scores: pl.DataFrame
    models: list[Trainer] | None


class Trainer(nn.Module):
    """Common training logic for models"""

    # Avoid Dynamo recompilations
    COMPILER_FRIENDLY = True

    def _epoch(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        opt: torch.optim.Optimizer,
        batch_size: int,
    ) -> float:
        total_loss = 0.0
        if self.COMPILER_FRIENDLY:
            last_idx = imgs.shape[0] - imgs.shape[0] % batch_size
            imgs, labels = imgs[:last_idx], labels[:last_idx]

        for batch_start in range(0, imgs.shape[0], batch_size):
            y = labels[batch_start : batch_start + batch_size]
            res = self(imgs[batch_start : batch_start + batch_size])
            loss = nn.functional.cross_entropy(res, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach().item()

        return total_loss

    def fit(
        self,
        data: Dataset | tuple[torch.Tensor, torch.Tensor],
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
        epoch_callback: Callable[[Self, float], None] | None = None,
        shuffle_seed: int = 0,
        run_seed: int = 1,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> Self:
        self.train().to("cuda")

        shuffle_gen = torch.Generator(device="cuda").manual_seed(shuffle_seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        if hasattr(data, "x_train"):
            imgs = torch.as_tensor(data.x_train, device="cuda")
            labels = torch.as_tensor(data.y_train, device="cuda")
        else:
            imgs, labels = data

        torch.manual_seed(run_seed)

        for _ in trange(
            epochs, desc="Training", unit="epoch", smoothing=0, disable=not verbose
        ):
            if shuffle:
                idxs = torch.randperm(
                    imgs.shape[0], generator=shuffle_gen, device="cuda"
                )
                imgs, labels = imgs[idxs], labels[idxs]

            train_loss = self._epoch(imgs, labels, opt, batch_size)
            if epoch_callback is not None:
                epoch_callback(self, train_loss)

        return self

    def evaluate(
        self,
        data: Dataset | tuple[torch.Tensor, torch.Tensor],
        batch_size: int = 10_000,
        label_names: list[str] | None = None,
    ) -> dict:
        with torch.set_grad_enabled(self.COMPILER_FRIENDLY):
            self.eval().to("cuda")
            if hasattr(data, "x_test"):
                imgs = torch.as_tensor(data.x_test, device="cuda")
                labels_np = data.y_test.numpy(force=True)
                label_names = data.label_names
            else:
                imgs, labels = data
                imgs = imgs.cuda()
                labels_np = labels.numpy(force=True)

            if self.COMPILER_FRIENDLY:
                last_idx = imgs.shape[0] - imgs.shape[0] % batch_size
                imgs, labels_np = imgs[:last_idx], labels_np[:last_idx]

            preds = [
                self(imgs[batch_start : batch_start + batch_size])
                .argmax(1)
                .numpy(force=True)
                for batch_start in range(0, imgs.shape[0], batch_size)
            ]
            preds = np.concat(preds)
            assert not np.isnan(preds).any()
            return classification_report(
                labels_np,
                preds,
                output_dict=True,
                target_names=label_names,
                zero_division=0,
            )

    @classmethod
    def fit_many(
        cls,
        data: Dataset,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.001,
        pool_fn: str | Callable[[int, dict], nn.Module] = "standard-2",
        init: dict[str, str | float] | None = None,
        seed: int = 0,
        debug: bool = False,
        count: int = 20,
        return_models: bool = False,
        epoch_callback: Callable[[Self, float], None] | None = None,
        description: str | None = None,
        progress_bar: bool = True,
        torch_compile_mode: str | None = "default",
        **init_kwargs,
    ) -> FitManyResult:
        if torch_compile_mode is not None:
            # noinspection PyProtectedMember
            torch._dynamo.reset()

        run_scores = []
        run_timings = []
        compile_timings = []
        model_seeds, run_seeds, shuffle_seeds = split_seed(count, seed, groups=3)

        bar = tqdm(
            zip(model_seeds, run_seeds, shuffle_seeds, strict=True),
            unit="run",
            desc=f"{pool_fn}:{init}" if description is None else description,
            total=count,
            disable=not progress_bar,
        )
        imgs, labels, test_imgs, test_labels = data.as_cuda(except_y_test=True)
        models = [] if return_models else None
        for m_seed, r_seed, s_seed in bar:
            model = cls(
                img_channels=data.img_channels,
                num_classes=data.num_classes,
                pool_fn=pool_fn,
                debug=debug,
                init=init,
                init_seed=m_seed,
                **init_kwargs,
            ).cuda()
            start_compile_time = time.perf_counter()

            # All the lazy modules need to be initialised before the compiler sees them.
            model(imgs[:batch_size])

            if torch_compile_mode is not None:
                model.compile(mode=torch_compile_mode)
                model(imgs[:batch_size])

            compile_timings.append(time.perf_counter() - start_compile_time)

            start_time = time.perf_counter()
            model.fit(
                (imgs, labels),
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                shuffle_seed=s_seed,
                run_seed=r_seed,
                verbose=False,
                epoch_callback=epoch_callback,
            )
            run_timings.append(time.perf_counter() - start_time)

            run_scores.append(
                model.evaluate(
                    (test_imgs, test_labels),
                    label_names=data.label_names,
                    batch_size=batch_size if cls.COMPILER_FRIENDLY else 10_000,
                )
            )
            bar.set_postfix(last_acc=run_scores[-1]["accuracy"])
            if return_models:
                models.append(model)

        scores = (
            reports_to_df(run_scores)
            .with_columns(
                train_times=pl.Series(run_timings),
                compile_times=pl.Series(compile_timings),
            )
            .with_columns(times=pl.col("train_times").add(pl.col("compile_times")))
        )
        return FitManyResult(scores, models)
