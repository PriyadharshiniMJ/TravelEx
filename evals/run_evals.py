"""Run the golden dataset through TravelEx and score it in LangSmith.

    python evals/run_evals.py                  # all 25 cases (about 25-40 minutes, real API costs)
    python evals/run_evals.py --ids A1 C4 E5   # a few cases
    python evals/run_evals.py --no-judges      # code checks only, no LLM grading

Each case is a full run with live search and the configured model. Results land in LangSmith as an
experiment on the dataset; group by metadata.focus to see each node's scores.
"""
import argparse
import os
import sys
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, evaluate

sys.path.insert(0, str(Path(__file__).parent))

import travelex  # noqa: E402,F401  (loads .env)
from evaluators import CODE_EVALUATORS, JUDGES, completion, critical, judge_model_problem, success  # noqa: E402
from target import run_case_safely  # noqa: E402
from upload_dataset import DATASET, stale_ids  # noqa: E402


PASS_MEAN = 0.8  # a case passes with this mean and no critical failure
HEADLINE = {"critical", "success"}  # reported on their own, not averaged with the section checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", help="only these case ids, e.g. A1 C4")
    parser.add_argument("--no-judges", action="store_true", help="skip the LLM-as-judge evaluators")
    parser.add_argument("--concurrency", type=int, default=1,  # each trip fires many searches at once: more
                        help="cases run at once (default 1: parallel trips hit the search service's rate limit)")
    args = parser.parse_args()

    if not args.no_judges and (problem := judge_model_problem()):
        sys.exit(f"{problem}\nOr run without the LLM judges: --no-judges")
    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET))
    if stale := stale_ids(examples):
        sys.exit(f"The LangSmith copy of {DATASET} differs from the local golden file for: {', '.join(stale)}.\n"
                 f"Upload it again first: python evals/upload_dataset.py --replace")
    if args.ids:
        wanted = {i.upper() for i in args.ids}
        examples = [e for e in examples if (e.metadata or {}).get("id") in wanted]
    if not examples:
        sys.exit(f"No examples to run. Upload first: python evals/upload_dataset.py")

    results = evaluate(lambda inputs: run_case_safely(inputs), data=examples,
                       evaluators=CODE_EVALUATORS + [completion, critical, success] + ([] if args.no_judges else JUDGES),
                       experiment_prefix="travelex", max_concurrency=args.concurrency,
                       metadata={"model": os.getenv("LLM_MODEL"), "cases": len(examples),
                                 "judge_model": None if args.no_judges else os.getenv("LLM_MODEL_JUDGE")})
    wait_for_all_tracers()

    print(f"\nExperiment: {results.experiment_name}")
    done_well = []
    for row in results:
        case = (row["example"].metadata or {}).get("id")
        scored = [r for r in row["evaluation_results"]["results"] if r.score is not None]
        severe = next((r for r in scored if r.key == "critical"), None)
        well = next((r for r in scored if r.key == "success"), None)
        done_well.append(bool(well and well.score == 1))
        rest = [r for r in scored if r.key not in HEADLINE]  # critical and success stay out of the mean
        weak = [f"{r.key} {r.score:.2f}: {r.comment}" for r in rest if r.score < 1]
        mean = sum(r.score for r in rest) / len(rest) if rest else 0
        passed = mean >= PASS_MEAN and (severe is None or severe.score == 1)
        print(f"{case}: {'PASS' if passed else 'FAIL'}, mean {mean:.2f} over {len(rest)} checks"
              + (f"; CRITICAL: {severe.comment}" if severe and severe.score < 1 else "")
              + (f"; not done well: {well.comment}" if well and well.score < 1 else "")
              + "".join(f"\n    {w}" for w in weak))
    if done_well:
        print(f"\nDone well (task completion): {sum(done_well)}/{len(done_well)} = {sum(done_well) / len(done_well):.0%}")


if __name__ == "__main__":
    main()
