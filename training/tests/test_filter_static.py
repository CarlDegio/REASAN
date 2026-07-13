from pathlib import Path


TRAIN_FILTER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_filter.py"


def test_filter_training_enables_cudnn_for_recurrent_policy():
    source = TRAIN_FILTER_SCRIPT.read_text()

    assert "torch.backends.cudnn.enabled = True" in source
