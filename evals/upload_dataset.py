"""Upload the golden dataset to LangSmith: python evals/upload_dataset.py [--replace]

Without --replace an existing dataset is left alone. --replace deletes the dataset's examples and uploads
them again (the dataset itself, and experiments run against it, are kept).
"""
import argparse
import json
from pathlib import Path

from langsmith import Client

import travelex  # noqa: F401  (loads .env)

GOLDEN = Path(__file__).parent / "golden" / "travelex_golden_v1.jsonl"
DATASET = "travelex-golden-v1"


def local_cases() -> dict[str, dict]:
    return {c["metadata"]["id"]: c for c in (json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()
                                             if line.strip())}


def stale_ids(examples) -> list[str]:
    """Cases whose LangSmith copy differs from the local golden file (or that one side lacks). Experiments read the
    LangSmith copy, so a stale one scores the agent against expectations the dataset no longer has."""
    local = local_cases()
    remote = {(e.metadata or {}).get("id"): e for e in examples}
    differ = [i for i in local if i not in remote or remote[i].inputs != local[i]["inputs"]
              or remote[i].outputs != local[i]["outputs"]]
    return sorted(differ + [i for i in remote if i not in local])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="re-upload the examples of an existing dataset")
    args = parser.parse_args()
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
        if not args.replace:
            print(f"{DATASET} already exists ({dataset.example_count} examples). Use --replace to re-upload.")
            return
        old = [e.id for e in client.list_examples(dataset_id=dataset.id)]
        if old:
            client.delete_examples(example_ids=old)
    else:
        dataset = client.create_dataset(DATASET, description="TravelEx golden cases: 25 trips, 5 per focus area. "
                                                             "Source: evals/golden/build_golden.py")
    client.create_examples(dataset_id=dataset.id, examples=cases)
    print(f"Uploaded {len(cases)} examples to {DATASET}.")


if __name__ == "__main__":
    main()
