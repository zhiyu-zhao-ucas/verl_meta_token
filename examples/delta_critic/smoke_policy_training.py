# Copyright 2026 Individual Contributor: zhiyu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check real policy microbatch equivalence, exact resume, and offline updates.

Use a completed round from smoke_policy_rounds as --source-round. Restrict visible
GPUs before launching; --two-gpu additionally compares one- and two-rank updates.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file


def weights(path):
    state = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        state.update(load_file(str(shard)))
    if not state:
        raise ValueError(f"No safetensors model found at {path}")
    return state


def compare(left, right, *, atol=2e-6, rtol=2e-5):
    left, right = weights(left), weights(right)
    assert left.keys() == right.keys()
    maximum = 0.0
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol)
        maximum = max(maximum, (left[key].float() - right[key].float()).abs().max().item())
    return maximum


def launch(source, output, name, *, microbatch=1, ranks=1, config=None, extra=()):
    checkpoints = sorted((source / "critic").glob("step_*/COMPLETE"))
    if not checkpoints:
        raise ValueError("Source round must have a completed critic")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={ranks}",
        "--module",
        "examples.delta_critic.policy_train",
        "--config",
        str(config or source / "policy.yaml"),
        "--rollouts",
        str(source / "data/rollouts_train_regen.jsonl"),
        "--labels",
        str(source / "data/mc_labels_train.jsonl"),
        "--split",
        str(source / "split.json"),
        "--critic",
        str(checkpoints[-1].parent),
        "--output",
        str(output / name),
        "--global-batch",
        "4",
        "--microbatch",
        str(microbatch),
        "--max-steps",
        "3",
        *extra,
    ]
    log = output / f"{name}.log"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    print(json.dumps({"test": name, "log": str(log)}), flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log}")
    return output / name / "final_hf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-round", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--two-gpu", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.source_round).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = launch(source, output, "baseline")
    reexported = launch(
        source,
        output,
        "reexport",
        extra=("--resume", str(output / "baseline/step_00000003")),
    )
    launch(source, output, "interrupted", extra=("--stop-after", "1"))
    resumed = launch(
        source,
        output,
        "resumed",
        extra=(
            "--resume",
            str(output / "interrupted/step_00000001"),
        ),
    )
    batched = launch(source, output, "micro2", microbatch=2)
    report = {
        "complete_checkpoint_export_max_abs": compare(baseline, reexported, atol=0, rtol=0),
        "resume_max_abs": compare(baseline, resumed, atol=0, rtol=0),
        # Changing the physical microbatch changes reduction order. Tiny
        # loss differences around 1e-6 can grow to ~4e-5 in Adam weights.
        "microbatch_max_abs": compare(baseline, batched, atol=5e-5, rtol=5e-3),
    }
    if args.two_gpu:
        distributed = launch(source, output, "two_gpu", ranks=2)
        report["two_gpu_max_abs"] = compare(baseline, distributed, atol=5e-5, rtol=5e-3)
    offline = yaml.safe_load((source / "policy.yaml").read_text())
    offline.update(
        mode="offline_legacy",
        behavior_logprob_source="stored",
        kl_reference="behavior",
        kl_estimator="source_sampled_reverse",
        kl_coef=0.01,
        window="legacy_tail",
    )
    offline.pop("reference_model_path", None)
    config = output / "offline.yaml"
    config.write_text(yaml.safe_dump(offline))
    trained = weights(launch(source, output, "offline", config=config))
    initial = weights(offline["model_path"])
    difference = max((initial[key].float() - trained[key].float()).abs().max().item() for key in initial)
    assert difference > 0, "Offline update did not change weights"
    report["offline_max_abs_change"] = difference
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
