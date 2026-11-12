import subprocess
import sys


def test_train_help_lists_every_option():
    """Fails if train.py is truncated or broken: --help must run and show the last option."""
    result = subprocess.run(
        [sys.executable, "-m", "skin_lesion_uncertainty.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--save-test-logits" in result.stdout
