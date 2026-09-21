import pytest

from benchmarks import kev


def test_server_identity_must_match_pinned_checkpoint():
    payload = {"models": [{"id": "kev-latest", "run": kev.RUN, "base": kev.BASE_MODEL}]}
    assert kev._validate_server(payload) == {"server_run": kev.RUN, "server_base": kev.BASE_MODEL}

    snapshot = f"/cache/models--jaredpalmer--kev-4b/snapshots/{kev.MODEL_REVISION}"
    payload["models"][0]["run"] = snapshot
    assert kev._validate_server(payload)["server_run"] == snapshot

    payload["models"][0]["run"] = "jaredpalmer/kev-4b"
    with pytest.raises(ValueError, match="expected"):
        kev._validate_server(payload)


def test_training_exposure_is_part_of_identity():
    assert kev.identity("banking77")["training_exposure"]["category"] == "benchmark_train_split"
    assert kev.identity("sst2")["training_exposure"]["category"] == "related_task"
    assert kev.identity("stsb")["training_exposure"]["category"] == "no_known_task_specific_training"
