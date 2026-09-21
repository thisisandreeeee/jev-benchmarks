from benchmarks import sst2
from benchmarks.providers import Inference, NoulResult


class FakeProvider:
    def infer(self, state, question):
        return Inference(NoulResult(0.9), {"provider": "fake"})


def test_evaluate_one_uses_noul_for_positive_sentiment():
    record = sst2.evaluate_one(
        FakeProvider(),
        7,
        {"sentence": "A wonderful film.", "label": 1},
    )
    assert record["dataset_id"] == 7
    assert record["state"] == "A wonderful film."
    assert record["question"]["instructions"] == sst2.INSTRUCTION
    assert record["question"]["type"] == "noul"
    assert record["expected"] == 1
    assert record["result"] == {"type": "noul", "noul": 0.9}


def test_typesafe_identity_pins_glue_configuration_and_validation_split():
    value = sst2.identity("typesafe")
    assert value["provider"] == "typesafe"
    assert value["model"] == sst2.JEV_MODEL
    assert value["schema_version"] == 2
    assert value["dataset"]["config"] == "sst2"
    assert value["dataset"]["split"] == "validation"
    assert value["positive_outcome"] == "positive"


def test_huggingface_identity_pins_model_and_same_evaluation_split():
    value = sst2.identity("huggingface")
    assert value["provider"] == "huggingface-local"
    assert value["model"] == sst2.MODEL
    assert value["schema_version"] == 1
    assert value["model_revision"] == sst2.MODEL_REVISION
    assert value["model_card"] == sst2.MODEL_CARD
    assert value["positive_label"] == sst2.POSITIVE_LABEL
    assert value["dataset"]["config"] == "sst2"
    assert value["dataset"]["split"] == "validation"


def test_provider_selects_existing_default_output_paths():
    jev = sst2.parse_args(["--provider", "typesafe"])
    huggingface = sst2.parse_args(["--provider", "huggingface"])
    assert jev.output == sst2.ROOT / "runs" / "sst2" / f"typesafe-{sst2.JEV_MODEL}"
    assert huggingface.output == sst2.ROOT / "runs" / "sst2" / f"huggingface-{sst2.SLUG}"
