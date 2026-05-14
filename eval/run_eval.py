"""Score a predictor against the public eval JSON.

Usage:
    python -m eval.run_eval                       # scores heuristic
    python -m eval.run_eval --predictor hybrid    # scores hybrid (uses LLM for ambiguous)
    python -m eval.run_eval --predictor llm_only  # full LLM (slow, expensive)
    python -m eval.run_eval --limit 100           # first 100 cases only

The harness accepts any callable that takes a list[Case] and returns list[Prediction]
in the same order/shape as the API contract. This way the same code paths used in
production are what we measure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import Case, Prediction  # noqa: E402

DEFAULT_JSON = ROOT / "relevant_priors_public.json"


def load_public(path: Path) -> tuple[list[Case], dict[tuple[str, str], bool]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = [Case.model_validate(c) for c in data["cases"]]
    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}
    return cases, truth


def score(
    cases: list[Case],
    truth: dict[tuple[str, str], bool],
    predictor: Callable[[list[Case]], list[Prediction]],
    show_errors: int = 20,
) -> dict:
    t0 = time.perf_counter()
    preds = predictor(cases)
    elapsed = time.perf_counter() - t0

    pred_map: dict[tuple[str, str], bool] = {}
    for p in preds:
        pred_map[(p.case_id, p.study_id)] = p.predicted_is_relevant

    # Score against truth — skipped predictions count as wrong.
    correct = 0
    incorrect = 0
    skipped = 0
    confusion: Counter = Counter()  # (true, pred) -> count
    error_log: list[tuple[str, str, bool, bool, str, str]] = []

    pair_to_descs: dict[tuple[str, str], tuple[str, str]] = {}
    for c in cases:
        for p in c.prior_studies:
            pair_to_descs[(c.case_id, p.study_id)] = (
                c.current_study.study_description,
                p.study_description,
            )

    for key, label in truth.items():
        if key not in pair_to_descs:
            continue  # truth pair not in eval cases (shouldn't happen)
        if key not in pred_map:
            skipped += 1
            incorrect += 1
            confusion[(label, "skipped")] += 1
            continue
        pred = pred_map[key]
        confusion[(label, pred)] += 1
        if pred == label:
            correct += 1
        else:
            incorrect += 1
            cd, pd = pair_to_descs[key]
            error_log.append((key[0], key[1], label, pred, cd, pd))

    total = correct + incorrect
    accuracy = correct / total if total else 0.0

    print(f"=== {predictor.__name__} on {len(cases)} cases ===")
    print(f"elapsed:    {elapsed:.2f}s ({elapsed * 1000 / max(total, 1):.2f} ms/pair)")
    print(f"pairs:      {total}")
    print(f"accuracy:   {accuracy:.4f}  ({correct}/{total})")
    print(f"skipped:    {skipped}")
    print()
    print("Confusion (true, pred): count")
    for (t, p), n in sorted(confusion.items()):
        print(f"  ({t!s:5}, {p!s:6}) {n}")

    if show_errors and error_log:
        # Group errors by description-pair to surface the highest-impact mistakes.
        by_pair: dict[tuple[str, str, bool, bool], int] = defaultdict(int)
        for _, _, label, pred, cd, pd in error_log:
            by_pair[(cd, pd, label, pred)] += 1
        print()
        print(f"Top {show_errors} error description-pairs:")
        for (cd, pd, label, pred), n in sorted(by_pair.items(), key=lambda x: -x[1])[:show_errors]:
            print(f"  x{n:3d}  truth={label!s:5} pred={pred!s:5}  CURR: {cd[:42]:42s} | PRIOR: {pd[:42]}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "elapsed": elapsed,
        "total_pairs": total,
        "confusion": dict(confusion),
    }


def make_predictor(name: str) -> Callable[[list[Case]], list[Prediction]]:
    if name == "always_false":
        def always_false(cases: list[Case]) -> list[Prediction]:
            return [
                Prediction(case_id=c.case_id, study_id=p.study_id, predicted_is_relevant=False)
                for c in cases for p in c.prior_studies
            ]
        return always_false

    if name == "heuristic":
        from app.heuristic import predict_pair as _pp

        def heuristic(cases: list[Case]) -> list[Prediction]:
            return [
                Prediction(
                    case_id=c.case_id,
                    study_id=p.study_id,
                    predicted_is_relevant=_pp(c.current_study.study_description, p.study_description).predicted,
                )
                for c in cases for p in c.prior_studies
            ]
        return heuristic

    if name == "hybrid":
        from app.classifier import predict_cases as _pc

        def hybrid(cases: list[Case]) -> list[Prediction]:
            return _pc(cases)
        return hybrid

    if name == "classifier":
        from app.classifier_model import predict_batch, is_available
        from app.parser import parse_description

        if not is_available():
            raise SystemExit("classifier model not loaded — run: python -m eval.train_classifier --save")

        def classifier(cases: list[Case]) -> list[Prediction]:
            pairs = []
            keys = []
            for c in cases:
                ct = parse_description(c.current_study.study_description)
                for p in c.prior_studies:
                    pt = parse_description(p.study_description)
                    pairs.append((
                        c.current_study.study_description,
                        p.study_description,
                        c.current_study.study_date,
                        p.study_date,
                        ct,
                        pt,
                    ))
                    keys.append((c.case_id, p.study_id))
            preds = predict_batch(pairs)
            return [
                Prediction(case_id=k[0], study_id=k[1], predicted_is_relevant=p)
                for k, p in zip(keys, preds)
            ]
        return classifier

    raise SystemExit(f"unknown predictor: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="heuristic", choices=["always_false", "heuristic", "hybrid", "classifier"])
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--limit", type=int, default=0, help="if >0, only score the first N cases")
    ap.add_argument("--errors", type=int, default=25)
    args = ap.parse_args()

    cases, truth = load_public(Path(args.json))
    if args.limit:
        cases = cases[: args.limit]
        eligible = {(c.case_id, p.study_id) for c in cases for p in c.prior_studies}
        truth = {k: v for k, v in truth.items() if k in eligible}

    score(cases, truth, make_predictor(args.predictor), show_errors=args.errors)


if __name__ == "__main__":
    main()
