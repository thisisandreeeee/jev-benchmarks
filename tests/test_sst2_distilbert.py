from benchmarks.sst2_distilbert import MODEL, MODEL_CARD, MODEL_REVISION, POSITIVE_LABEL, identity


def test_identity_pins_model_and_same_sst2_evaluation_split():
    value = identity()
    assert value["provider"] == "huggingface-local"
    assert value["model"] == MODEL
    assert value["model_revision"] == MODEL_REVISION
    assert value["model_card"] == MODEL_CARD
    assert value["positive_label"] == POSITIVE_LABEL
    assert value["dataset"]["config"] == "sst2"
    assert value["dataset"]["split"] == "validation"
