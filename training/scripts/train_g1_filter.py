"""Train the migrated Unitree G1 REASEN safety filter."""

import sys

if "--task" not in sys.argv:
    sys.argv.extend(("--task", "Unitree-G1-Filter"))

from train_filter import main


if __name__ == "__main__":
    main()
