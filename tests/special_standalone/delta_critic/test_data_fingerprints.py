# Copyright 2026 Individual Contributor: zhiyu
# Licensed under the Apache License, Version 2.0

import pytest

from examples.delta_critic.data_fingerprints import (
    DENIED_DATA_FINGERPRINTS,
    reject_denied_data_fingerprints,
)


def test_contaminated_label_fingerprint_is_rejected():
    digest = "389ad83f26e05ff73fab4992484cf309d119a6ad0d60ca1e7e48f2770c2a4be9"
    assert digest in DENIED_DATA_FINGERPRINTS
    with pytest.raises(ValueError, match="threaded math-verify"):
        reject_denied_data_fingerprints({"labels": digest})


def test_fingerprint_override_is_explicitly_audit_only():
    digest = next(iter(DENIED_DATA_FINGERPRINTS))
    reject_denied_data_fingerprints({"labels": digest}, allow_for_audit=True)
    reject_denied_data_fingerprints({"labels": "0" * 64})


def test_fingerprint_mapping_is_validated():
    with pytest.raises(ValueError, match="mapping"):
        reject_denied_data_fingerprints(["not", "a", "mapping"])
    with pytest.raises(ValueError, match="must be a string"):
        reject_denied_data_fingerprints({"labels": 1})
