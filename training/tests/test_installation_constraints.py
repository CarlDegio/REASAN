from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_SOURCE = REPO_ROOT / "training" / "IsaacLab" / "source"


def test_readme_installs_isaaclab_without_unused_rl_frameworks():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "./isaaclab.sh --install none" in readme
    assert "./isaaclab.sh --install\n" not in readme


def test_isaaclab_packages_pin_the_cu128_pytorch_builds():
    setup_files = [
        ISAACLAB_SOURCE / "isaaclab" / "setup.py",
        ISAACLAB_SOURCE / "isaaclab_rl" / "setup.py",
        ISAACLAB_SOURCE / "isaaclab_tasks" / "setup.py",
    ]

    for setup_file in setup_files:
        source = setup_file.read_text()
        assert '"torch==2.7.0+cu128"' in source, setup_file
        assert '"torchvision==0.22.0+cu128"' in source, setup_file
