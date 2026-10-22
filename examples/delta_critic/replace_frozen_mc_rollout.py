# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0
"""Replace one equal-state-count rollout in an immutable frozen MC snapshot."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .freeze_completed_mc_subset import _STATE_ID_PREFIX

FILES = {
    "rollouts": "rollouts_train_regen.jsonl",
    "states": "states_train.jsonl",
    "labels": "mc_labels_collector.jsonl",
    "continuations": "continuations_collector.jsonl",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                yield raw, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc


def _continuation_state_id(raw: bytes, path: Path, line_number: int) -> str:
    match = _STATE_ID_PREFIX.match(raw, 0, min(len(raw), 256))
    if match is not None:
        return match.group(1).decode("utf-8")
    try:
        return str(json.loads(raw)["state_id"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Invalid continuation identity {path}:{line_number}") from exc


def replace(
    base: Path,
    replacement: Path,
    output: Path,
    drop_rollout_id: str,
    reason: str = "strict terminal scorer timed out after declared retry",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    base_manifest = json.loads((base / "source_manifest.json").read_text())
    replacement_manifest = json.loads((replacement / "source_manifest.json").read_text())
    expected_status = "frozen_collector_snapshot_requires_strict_rescoring"
    if base_manifest.get("status") != expected_status or replacement_manifest.get("status") != expected_status:
        raise ValueError("Both inputs must be frozen, unrescored collector snapshots")
    base_plan = json.loads((base / "selection_manifest.json").read_text())
    replacement_plan = json.loads((replacement / "selection_manifest.json").read_text())
    dropped = [row for row in base_plan["selection"]["selected"] if row["rollout_id"] == drop_rollout_id]
    added = replacement_plan["selection"]["selected"]
    if len(dropped) != 1 or len(added) != 1:
        raise ValueError("Replacement requires exactly one dropped and one added rollout")
    dropped, added = dropped[0], added[0]
    if dropped["state_count"] != added["state_count"]:
        raise ValueError("Replacement must preserve the selected-state count")
    drop_states, add_states = set(dropped["state_ids"]), set(added["state_ids"])
    if drop_states & add_states or drop_rollout_id == added["rollout_id"]:
        raise ValueError("Replacement identities overlap the dropped population")
    split = json.loads((base / "split.json").read_text())
    role = split.pop(drop_rollout_id)
    if added["rollout_id"] in split:
        raise ValueError("Replacement rollout already exists in the base split")
    split[added["rollout_id"]] = role

    output.mkdir(parents=True)
    final = {name: output / filename for name, filename in FILES.items()}
    final.update(
        {
            "split": output / "split.json",
            "selection": output / "selection_manifest.json",
            "manifest": output / "source_manifest.json",
        }
    )
    temporary = {name: path.with_name("." + path.name + ".incomplete") for name, path in final.items()}
    counts = Counter()
    try:
        for name, filename in FILES.items():
            with temporary[name].open("wb") as destination:
                if name == "continuations":
                    for source, allowed, dropping in (
                        (base / filename, None, drop_states),
                        (replacement / filename, add_states, set()),
                    ):
                        with source.open("rb") as stream:
                            for line_number, raw in enumerate(stream, start=1):
                                if not raw.strip():
                                    continue
                                identity = _continuation_state_id(raw, source, line_number)
                                if identity in dropping:
                                    continue
                                if allowed is not None and identity not in allowed:
                                    raise ValueError("Unexpected state in replacement continuations")
                                destination.write(raw)
                                counts[name] += 1
                else:
                    for raw, row in _rows(base / filename):
                        identity = str(row["id"]) if name == "rollouts" else str(row["state_id"])
                        should_drop = identity == drop_rollout_id if name == "rollouts" else identity in drop_states
                        if should_drop:
                            continue
                        destination.write(raw)
                        counts[name] += 1
                    for raw, row in _rows(replacement / filename):
                        identity = str(row["id"]) if name == "rollouts" else str(row["state_id"])
                        expected = added["rollout_id"] if name == "rollouts" else None
                        if name == "rollouts" and identity != expected:
                            raise ValueError("Unexpected rollout in replacement snapshot")
                        if name != "rollouts" and identity not in add_states:
                            raise ValueError("Unexpected state in replacement snapshot")
                        destination.write(raw)
                        counts[name] += 1
        expected_counts = base_manifest["counts"]
        if any(counts[name] != int(expected_counts[name]) for name in FILES):
            raise ValueError(f"Replacement changed population counts: {dict(counts)}")

        updated_plan = json.loads(json.dumps(base_plan))
        updated_plan["selection"]["selected"] = [
            added if row["rollout_id"] == drop_rollout_id else row for row in updated_plan["selection"]["selected"]
        ]
        updated_plan["split"]["mapping"] = split
        updated_plan["population_replacement"] = {
            "reason": reason,
            "dropped": dropped,
            "added": added,
            "preserved_split_role": role,
        }
        temporary["split"].write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        temporary["selection"].write_text(json.dumps(updated_plan, indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema_version": 1,
            "status": expected_status,
            "counts": dict(expected_counts),
            "split": dict(base_manifest["split"]),
            "population_replacement": {
                "reason": reason,
                "dropped_rollout_id": drop_rollout_id,
                "added_rollout_id": added["rollout_id"],
                "state_count": added["state_count"],
                "split_role": role,
                "base_manifest_sha256": _hash(base / "source_manifest.json"),
                "replacement_manifest_sha256": _hash(replacement / "source_manifest.json"),
            },
        }
        manifest["outputs"] = {
            name: {"path": final[name].name, "sha256": _hash(temporary[name])}
            for name in (*FILES, "split", "selection")
        }
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name in (*FILES, "split", "selection", "manifest"):
            temporary[name].replace(final[name])
    except Exception:
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--drop-rollout-id", required=True)
    parser.add_argument(
        "--reason",
        default="strict terminal scorer timed out after declared retry",
        help="Auditable reason for replacing the rollout",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replace(
        Path(args.base),
        Path(args.replacement),
        Path(args.output),
        args.drop_rollout_id,
        args.reason,
    )
    print("DELTA_MC_REPLACE_RESULT=" + json.dumps({"output": args.output, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
