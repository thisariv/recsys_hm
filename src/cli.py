"""Командная строка для подготовки данных и запуска экспериментов."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.tracking import init_mlflow

COMMANDS = ["prep", "split", "atomic", "atomic-lgcn", "train", "eval", "report"]


def _run_prep(config: dict[str, Any]) -> None:
    """Скачать и подготовить исходные данные."""
    import mlflow

    from src.data.download import ensure_raw_data
    from src.data.prepare import prepare

    data_cfg = config["data"]
    ensure_raw_data(
        competition=data_cfg["competition"],
        files=data_cfg["files"],
        raw_dir=Path(data_cfg["raw_dir"]),
    )

    stats = prepare(config)

    init_mlflow(config)
    with mlflow.start_run(run_name="prep"):
        mlflow.set_tag("stage", "prep")
        mlflow.log_params(
            {
                "seed": config["seed"],
                "last_n_months": data_cfg["sample"]["last_n_months"],
                "max_users": data_cfg["sample"]["max_users"],
                "min_user_interactions": data_cfg["kcore"]["min_user_interactions"],
                "min_item_interactions": data_cfg["kcore"]["min_item_interactions"],
            }
        )
        mlflow.log_metrics(
            {
                "n_users": stats["n_users"],
                "n_items": stats["n_items"],
                "n_interactions": stats["n_interactions"],
                "density": stats["density"],
                "seq_len_median": stats["seq_len"]["median"],
                "seq_len_mean": stats["seq_len"]["mean"],
            }
        )
    print(
        f"[prep] done: n_users={stats['n_users']:,} "
        f"n_items={stats['n_items']:,} n_interactions={stats['n_interactions']:,}"
    )


def _run_split(config: dict[str, Any]) -> None:
    """Построить temporal leave-last-out split."""
    import json as _json

    import mlflow
    import pandas as pd

    from src.data.split import make_split

    processed = Path(config["paths"]["processed"])
    split_dir = Path(config["paths"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    min_days = config["split"]["min_days"]
    red_flag_frac = config["split"]["red_flag_repeat_drop_frac"]

    interactions = pd.read_parquet(processed / "interactions.parquet")
    print(f"[split] interactions: {len(interactions):,} rows, "
          f"{interactions['user_id'].nunique():,} users")

    train, holdouts, catalog, stats = make_split(interactions, min_days=min_days)
    stats["seed"] = config["seed"]

    train.to_parquet(split_dir / "train.parquet", index=False)
    holdouts.to_parquet(split_dir / "holdouts.parquet", index=False)
    pd.DataFrame({"item_id": catalog}).to_parquet(split_dir / "catalog.parquet", index=False)
    with (split_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        _json.dump(stats, f, indent=2, ensure_ascii=False)

    f_test = stats["funnel"]["test"]
    f_val = stats["funnel"]["val"]
    print(
        f"[split] n_ge3={stats['funnel']['n_users_ge3_days']:,} "
        f"n_test_eval={f_test['n_test_eval']:,} n_val_eval={f_val['n_val_eval']:,} "
        f"|V_train|={stats['n_v_train']:,}"
    )
    print(f"[split] catalog_hash={stats['catalog_hash']}")

    if stats["red_flag_repeat_drop"]:
        print(
            f"[split] высокая доля повторных test-таргетов: "
            f"{stats['repeat_drop_frac_test']:.1%} > {red_flag_frac:.0%} — "
            f"стоит проверить правила сплита."
        )

    init_mlflow(config)
    with mlflow.start_run(run_name="split"):
        mlflow.set_tag("stage", "split")
        mlflow.log_params(
            {
                "seed": config["seed"],
                "min_days": min_days,
                "strategy": config["split"]["strategy"],
                "catalog_hash": stats["catalog_hash"],
            }
        )
        mlflow.log_metrics(
            {
                "n_users_total": stats["funnel"]["n_users_total"],
                "n_users_ge3_days": stats["funnel"]["n_users_ge3_days"],
                "n_test_eval": f_test["n_test_eval"],
                "n_val_eval": f_val["n_val_eval"],
                "n_dropped_cold_test": f_test["n_dropped_cold"],
                "n_dropped_repeat_test": f_test["n_dropped_repeat"],
                "n_dropped_cold_val": f_val["n_dropped_cold"],
                "n_dropped_repeat_val": f_val["n_dropped_repeat"],
                "n_v_train": stats["n_v_train"],
                "n_interactions_train": stats["n_interactions_train"],
                "repeat_drop_frac_test": stats["repeat_drop_frac_test"],
                "train_prefix_len_median": stats["train_prefix_len"]["median"],
                "red_flag_repeat_drop": int(stats["red_flag_repeat_drop"]),
            }
        )


def _fit_or_load_cached(model: str, scorer_factory, cache_path: Path, loader, meta: dict):
    """Загрузить модель из кэша или обучить её заново."""
    import json as _json

    meta_path = cache_path.with_suffix(".meta.json")
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            cached_meta = _json.load(f)
        if cached_meta == meta:
            print(f"[eval] {model}: загружаю fit-артефакт из кэша {cache_path.name}")
            return loader()
    print(f"[eval] {model}: обучаю (кэш отсутствует/устарел)")
    scorer = scorer_factory()
    scorer.save(cache_path, meta)
    return scorer


def _build_scorer(config: dict[str, Any], model: str):
    """Создать скорер по имени модели."""
    import pandas as pd

    from src.data.split import catalog_hash

    split_dir = Path(config["paths"]["split_dir"])
    catalog = pd.read_parquet(split_dir / "catalog.parquet")["item_id"].to_numpy()
    models_dir = Path(config["paths"]["metrics_dir"]).parent / "models"

    if model == "random":
        from src.eval.scorers import RandomScorer

        return RandomScorer(n_items=len(catalog), seed=config["seed"])

    if model in ("popularity", "popularity_decay"):
        from src.models.popularity import PopularityScorer

        train = pd.read_parquet(split_dir / "train.parquet")
        half_life = config["models"]["popularity"]["half_life_days"]
        return PopularityScorer(variant=model, half_life_days=half_life).fit(train, catalog)

    chash = catalog_hash(catalog)

    if model == "itemknn":
        from src.models.itemknn import ItemKNNScorer

        hp = config["models"]["itemknn"]
        meta = {"catalog_hash": chash, "k_neighbors": hp["k_neighbors"],
                "aggregation": hp["aggregation"]}
        cache = models_dir / "itemknn"
        train = pd.read_parquet(split_dir / "train.parquet")
        return _fit_or_load_cached(
            model,
            lambda: ItemKNNScorer(hp["k_neighbors"], hp["aggregation"]).fit(train, catalog),
            cache,
            lambda: ItemKNNScorer.load(cache, catalog, hp["k_neighbors"], hp["aggregation"]),
            meta,
        )

    if model == "als":
        from src.models.als import ALSScorer

        hp = config["models"]["als"]
        meta = {"catalog_hash": chash, "seed": config["seed"], **hp}
        cache = models_dir / "als"
        train = pd.read_parquet(split_dir / "train.parquet")
        return _fit_or_load_cached(
            model,
            lambda: ALSScorer(hp["factors"], hp["regularization"], hp["iterations"],
                              hp["alpha"], config["seed"]).fit(train, catalog),
            cache,
            lambda: ALSScorer.load(cache, catalog, factors=hp["factors"],
                                   regularization=hp["regularization"],
                                   iterations=hp["iterations"], alpha=hp["alpha"],
                                   seed=config["seed"]),
            meta,
        )

    if model in ("sasrec", "gru4rec", "bert4rec"):
        from src.sasrec.scorer import load_real_sasrec, verify_bridge

        ckpt_dir = Path(f"artifacts/{model}/checkpoint")
        if not (ckpt_dir / "token_map.json").exists():
            raise RuntimeError(
                f"{model}-чекпойнт не найден в {ckpt_dir}/. Обучи на Kaggle (RecBole, тот же "
                f"sequential-формат atomic) и положи *.pth + token_map.json + meta.json."
            )
        scorer = load_real_sasrec(ckpt_dir, catalog, config, expected_hash=chash)

        import json as _json

        atomic_meta = None
        meta_path = Path("artifacts/sasrec/atomic/atomic_meta.json")
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                atomic_meta = _json.load(f)
        train = pd.read_parquet(split_dir / "train.parquet")
        info = verify_bridge(scorer, catalog, train, chash, atomic_meta)
        print(f"[{model}] словарь и каталог согласованы: {info}")
        return scorer

    if model == "bpr":
        from src.models.bpr import BPRScorer

        hp = config["models"]["bpr"]
        meta = {"catalog_hash": chash, "seed": config["seed"], **hp}
        cache = models_dir / "bpr"
        train = pd.read_parquet(split_dir / "train.parquet")
        return _fit_or_load_cached(
            model,
            lambda: BPRScorer(hp["factors"], hp["regularization"], hp["iterations"],
                              hp["learning_rate"], config["seed"]).fit(train, catalog),
            cache,
            lambda: BPRScorer.load(cache, catalog, factors=hp["factors"],
                                   regularization=hp["regularization"], iterations=hp["iterations"],
                                   learning_rate=hp["learning_rate"], seed=config["seed"]),
            meta,
        )

    if model == "ease":
        from src.models.ease import load_real_ease

        ease_dir = Path("artifacts/ease/checkpoint")
        if not (ease_dir / "ease_B_topk.npz").exists() and not (ease_dir / "ease_B.npy").exists():
            raise RuntimeError(
                "EASE-веса не найдены в artifacts/ease/. Посчитай их на Kaggle и положи "
                "ease_B_topk.npz (или ease_B.npy) + meta.json."
            )
        scorer = load_real_ease(ease_dir, catalog, expected_hash=chash)
        assert scorer.n_items == len(catalog)
        print(f"[ease] загружен B ({'sparse' if scorer._sparse else 'dense'}), |V|={scorer.n_items}")
        return scorer

    if model == "multivae":
        from src.models.multivae import load_real_multivae

        ckpt_dir = Path("artifacts/multivae/checkpoint")
        if not (ckpt_dir / "token_map.json").exists():
            raise RuntimeError(
                "MultiVAE-чекпойнт не найден в artifacts/multivae/checkpoint/. Обучи на Kaggle "
                "(general recommender, плоские atomic recsys-hm-lightgcn) и положи *.pth + "
                "token_map.json + meta.json."
            )
        scorer = load_real_multivae(ckpt_dir, catalog, expected_hash=chash)
        assert scorer.n_items == len(catalog)
        assert scorer.n_missing / len(catalog) < 0.01, f"много missing: {scorer.n_missing}"
        print(f"[multivae] загружен: n_missing={scorer.n_missing}, n_recbole={scorer.n_recbole}, "
              f"enc-слоёв={len(scorer.enc)}, dec-слоёв={len(scorer.dec)}")
        return scorer

    if model == "lightgcn":
        from src.models.lightgcn import load_real_lightgcn

        ckpt_dir = Path("artifacts/lightgcn/checkpoint")
        if not (ckpt_dir / "user_emb.npy").exists():
            raise RuntimeError(
                "LightGCN-чекпойнт не найден в artifacts/lightgcn/checkpoint/. "
                "Обучи модель на Kaggle и положи user_emb.npy + item_emb.npy + "
                "user_token_map.json + item_token_map.json + meta.json."
            )
        scorer = load_real_lightgcn(ckpt_dir, catalog, expected_hash=chash)
        assert len(set(scorer.perm[~scorer.missing].tolist())) == int((~scorer.missing).sum())
        assert scorer.n_missing / len(catalog) < 0.01, f"много missing: {scorer.n_missing}"
        print(f"[lightgcn] мост ✓: n_missing={scorer.n_missing}, "
              f"user_emb={scorer.user_emb.shape}, item_emb={scorer.item_emb.shape}")
        return scorer

    raise ValueError(
        f"неизвестная model={model!r} (доступно: random, popularity, popularity_decay, "
        f"itemknn, als, bpr, canary, sasrec, gru4rec, bert4rec, lightgcn, ease, multivae)"
    )


def _persist_eval(result: dict[str, Any], config: dict[str, Any], split: str,
                  to_summary: bool = True) -> None:
    """Сохранить метрики модели и обновить общий summary."""
    import json as _json

    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = result["model"]
    with (metrics_dir / f"{model}_{split}.json").open("w", encoding="utf-8") as f:
        _json.dump(result, f, indent=2, ensure_ascii=False)

    if not to_summary:
        return

    summary_path = metrics_dir / f"summary_{split}.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = _json.load(f)
    cols = ["recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr", "n_cases"]
    summary[model] = {c: result[c] for c in cols}
    with summary_path.open("w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False)


def _model_hyperparams(config: dict[str, Any], model: str) -> dict[str, Any]:
    """Гиперпараметры модели для логирования в MLflow."""
    m = config.get("models", {})
    if model in ("popularity", "popularity_decay"):
        return {"half_life_days": m["popularity"]["half_life_days"]}
    if model == "itemknn":
        return dict(m["itemknn"])
    if model == "als":
        return dict(m["als"])
    if model == "bpr":
        return dict(m["bpr"])
    if model == "sasrec":
        return {k: v for k, v in m["sasrec"].items() if k != "recbole"}
    return {}


def _run_atomic(config: dict[str, Any]) -> None:
    """Сгенерировать sequence-файлы для RecBole."""
    import json as _json

    import pandas as pd

    from src.data.split import catalog_hash
    from src.sasrec.atomic import (
        assert_no_leak, gen_test_inter, gen_train_inter, gen_valid_inter, write_inter,
    )

    split_dir = Path(config["paths"]["split_dir"])
    out_dir = Path("artifacts/sasrec/atomic")

    train = pd.read_parquet(split_dir / "train.parquet")
    holdouts = pd.read_parquet(split_dir / "holdouts.parquet")
    catalog = pd.read_parquet(split_dir / "catalog.parquet")["item_id"].to_numpy()

    mode = config["models"]["sasrec"].get("transition_mode", "next_day")
    print(f"[atomic] генерирую train.inter (mode={mode}) ...")
    train_inter = gen_train_inter(train, mode=mode)
    valid_inter = gen_valid_inter(train, holdouts)
    test_inter = gen_test_inter(train, holdouts)

    print("[atomic] проверяю train и holdout-таргеты ...")
    leak_info = assert_no_leak(train_inter, train, holdouts, mode=mode)

    shas = {
        "hm.train.inter": write_inter(train_inter, out_dir / "hm.train.inter"),
        "hm.valid.inter": write_inter(valid_inter, out_dir / "hm.valid.inter"),
        "hm.test.inter": write_inter(test_inter, out_dir / "hm.test.inter"),
    }
    meta = {
        "catalog_hash": catalog_hash(catalog),
        "transition_mode": mode,
        "n_train_inter_rows": leak_info["n_train_inter_rows"],
        "expected_rows": leak_info["expected_rows"],
        "n_valid_rows": int(len(valid_inter)),
        "max_seq_len": config["models"]["sasrec"]["max_seq_len"],
        "sha256": shas,
    }
    with (out_dir / "atomic_meta.json").open("w", encoding="utf-8") as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[atomic] train.inter rows={meta['n_train_inter_rows']:,} "
          f"(ожидалось {meta['expected_rows']:,})")
    for name, sha in shas.items():
        print(f"[atomic] {name}: sha256={sha}")
    print(f"[atomic] -> {out_dir}/")


def _run_atomic_lgcn(config: dict[str, Any]) -> None:
    """Сгенерировать interaction-файлы для LightGCN."""
    import json as _json

    import pandas as pd

    from src.data.split import catalog_hash
    from src.models.lightgcn import gen_interactions, gen_valid, write_inter

    split_dir = Path(config["paths"]["split_dir"])
    out_dir = Path("artifacts/lightgcn/atomic")
    train = pd.read_parquet(split_dir / "train.parquet")
    holdouts = pd.read_parquet(split_dir / "holdouts.parquet")
    catalog = pd.read_parquet(split_dir / "catalog.parquet")["item_id"].to_numpy()

    inter = gen_interactions(train)
    valid = gen_valid(holdouts)
    print(f"[atomic-lgcn] train рёбер={len(inter):,} valid={len(valid):,}")

    shas = {
        "hm.train.inter": write_inter(inter, out_dir / "hm.train.inter"),
        "hm.valid.inter": write_inter(valid, out_dir / "hm.valid.inter"),
        "hm.test.inter": write_inter(valid, out_dir / "hm.test.inter"),
    }
    meta = {"catalog_hash": catalog_hash(catalog), "n_train_edges": int(len(inter)),
            "n_valid": int(len(valid)), "sha256": shas}
    with (out_dir / "atomic_meta.json").open("w", encoding="utf-8") as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)
    for name, sha in shas.items():
        print(f"[atomic-lgcn] {name}: sha256={sha}")
    print(f"[atomic-lgcn] -> {out_dir}/")


def _run_eval(config: dict[str, Any], split: str, model: str) -> None:
    """Запустить общую оценку выбранной модели."""
    from src.eval.harness import evaluate

    if model == "canary":
        from src.sasrec.canary import run_canary

        result = run_canary(config, split)
        # Canary проверяет адаптер и не является результатом модели.
        _persist_eval(result, config, split, to_summary=False)
        return

    scorer = _build_scorer(config, model)
    result = evaluate(
        scorer, split, config, log_mlflow=True, model_name=model,
        extra_params=_model_hyperparams(config, model),
    )
    _persist_eval(result, config, split)


def run_command(command: str, config_path: str, split: str = "val", model: str = "random") -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    if command == "prep":
        _run_prep(config)
        return

    if command == "split":
        _run_split(config)
        return

    if command == "atomic":
        _run_atomic(config)
        return

    if command == "atomic-lgcn":
        _run_atomic_lgcn(config)
        return

    if command == "eval":
        _run_eval(config, split=split, model=model)
        return

    print(f"[{command}] resolved config from {config_path}:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[{command}] command is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-hm",
        description="Data preparation and evaluation for the H&M recommender project",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in COMMANDS:
        p = sub.add_parser(cmd, help=cmd)
        p.add_argument("--config", default="configs/base.yaml", help="path to yaml config")
        if cmd == "eval":
            p.add_argument("--split", choices=["val", "test"], default="val",
                           help="split for evaluation")
            p.add_argument("--model", default="random",
                           help="model name")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_command(
        args.command,
        args.config,
        split=getattr(args, "split", "val"),
        model=getattr(args, "model", "random"),
    )


if __name__ == "__main__":
    main()
