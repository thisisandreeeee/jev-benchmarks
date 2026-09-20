from benchmarks.providers import Inference, NoulResult
from benchmarks.sst2 import INSTRUCTION, evaluate_one, identity


class FakeProvider:
    def infer(self, state, question):
        return Inference(NoulResult(0.9), {"provider": "fake"})


def test_evaluate_one_uses_noul_for_positive_sentiment():
    record = evaluate_one(
        FakeProvider(),
        7,
        {"sentence": "A wonderful film.", "label": 1},
    )
    assert record["dataset_id"] == 7
    assert record["state"] == "A wonderful film."
    assert record["question"]["instructions"] == INSTRUCTION
    assert record["question"]["type"] == "noul"
    assert record["expected"] == 1
    assert record["result"] == {"type": "noul", "noul": 0.9}


def test_identity_pins_glue_configuration_and_validation_split():
    value = identity()
    assert value["dataset"]["config"] == "sst2"
    assert value["dataset"]["split"] == "validation"
    assert value["positive_outcome"] == "positive"
