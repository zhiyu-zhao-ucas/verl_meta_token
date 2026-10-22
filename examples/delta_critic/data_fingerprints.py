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
"""Fail-closed registry for data proven invalid after collection."""

from collections.abc import Mapping

DENIED_DATA_FINGERPRINTS = {
    "389ad83f26e05ff73fab4992484cf309d119a6ad0d60ca1e7e48f2770c2a4be9": (
        "legacy threaded math-verify scorer silently converted failures to zero"
    ),
}


def reject_denied_data_fingerprints(
    fingerprints: Mapping[str, str] | None,
    *,
    allow_for_audit: bool = False,
) -> None:
    """Reject contaminated inputs; the override is for read-only audit tools."""
    if fingerprints is None:
        return
    if not isinstance(fingerprints, Mapping):
        raise ValueError("data_fingerprints must be a mapping")
    matches = []
    for name, digest in fingerprints.items():
        if not isinstance(digest, str):
            raise ValueError(f"Data fingerprint for {name} must be a string")
        if digest in DENIED_DATA_FINGERPRINTS:
            matches.append((str(name), digest, DENIED_DATA_FINGERPRINTS[digest]))
    if matches and not allow_for_audit:
        details = "; ".join(f"{name}={digest}: {reason}" for name, digest, reason in matches)
        raise ValueError(f"Refusing denied delta-critic data fingerprint(s): {details}")
