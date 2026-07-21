from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from run_amm_mnist import main as run_main

    default_config = str(ROOT / "configs" / "amm_cifar10.yaml")
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", default_config])
    return run_main()


if __name__ == "__main__":
    raise SystemExit(main())
