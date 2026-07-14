from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]


<<<<<<< HEAD
def test_filter_training_enables_cudnn_for_recurrent_policy():
    source = TRAIN_FILTER_SCRIPT.read_text()

    assert "torch.backends.cudnn.enabled = True" in source
=======
def test_training_entrypoints_do_not_disable_cudnn():
    entrypoints = list((TRAINING_DIR / "scripts").glob("*.py"))
    entrypoints += list((TRAINING_DIR / "tests").glob("manual_*.py"))

    offenders = [
        path.relative_to(TRAINING_DIR)
        for path in entrypoints
        if "torch.backends.cudnn.enabled = False" in path.read_text()
    ]

    assert offenders == []
>>>>>>> 9116d5b (add version constrain, add cudnn)
