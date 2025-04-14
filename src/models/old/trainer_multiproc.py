from __future__ import annotations

from torch import multiprocessing

# For reference: an experiment to try and use more of the GPU at smaller batch sizes
# Didn't end up being efficient, sadly: couldn't get more GPU utilisation.


def fit_many_par(
    cls,
    data: Dataset,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    pool_fn: str = "standard-2",
    init: str | float = 3.0,
    seed: int = 0,
    debug: bool = False,
    count: int = 20,
    *,
    workers: int,
    **init_kwargs,
) -> list[dict]:
    ctx = multiprocessing.get_context("spawn")
    model_seeds, run_seeds, shuffle_seeds = split_seed(count, seed, groups=3)

    train_imgs, train_labels, test_imgs, test_labels = data.as_cuda()
    test_labels = test_labels.cpu()

    with ctx.Pool(workers) as pool:
        tasks = (
            _ParFitArgs(
                worker_id=worker_id,
                cls=cls,
                m_seed=m_seed,
                pool_fn=pool_fn,
                init=init,
                train_imgs=train_imgs,
                train_labels=train_labels,
                test_imgs=test_imgs,
                test_labels=test_labels,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                r_seed=r_seed,
                s_seed=s_seed,
                img_channels=data.img_channels,
                num_classes=data.num_classes,
                debug=debug,
                init_kwargs=init_kwargs,
            )
            for worker_id, (m_seed, r_seed, s_seed) in enumerate(
                zip(model_seeds, run_seeds, shuffle_seeds, strict=True)
            )
        )

        bar = tqdm(
            pool.imap_unordered(cls._par_fit_entry, tasks, count // workers),
            unit="run",
            desc=f"Par: {pool_fn}-{init}",
            total=count,
            smoothing=0,
        )
        run_scores: list[dict | None] = [None] * count
        for worker_id, run in bar:
            run_scores[worker_id] = run
            bar.set_postfix(last_acc=run["accuracy"])

    assert all(run_scores), f"One of the runs failed? {run_scores=}"
    return run_scores


def _par_fit_entry(args: _ParFitArgs) -> tuple[int, dict]:
    torch.manual_seed(args.m_seed)
    return args.worker_id, (
        args.cls(
            img_channels=args.img_channels,
            num_classes=args.num_classes,
            pool_fn=args.pool_fn,
            debug=args.debug,
            init=args.init,
        )
        .fit(
            (args.train_imgs, args.train_labels),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            shuffle_seed=args.s_seed,
            run_seed=args.r_seed,
            verbose=False,
        )
        .evaluate((args.test_imgs, args.test_labels))
    )


class _ParFitArgs(typing.NamedTuple):
    worker_id: int
    cls: type[Trainer]
    train_imgs: torch.Tensor
    train_labels: torch.Tensor
    test_imgs: torch.Tensor
    test_labels: torch.Tensor
    img_channels: int
    num_classes: int
    pool_fn: str
    init: str | float
    debug: bool
    init_kwargs: dict[str, typing.Any]
    epochs: int
    batch_size: int
    lr: float
    m_seed: int
    r_seed: int
    s_seed: int
