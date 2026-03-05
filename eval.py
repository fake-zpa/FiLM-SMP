import argparse
from pathlib import Path

from lbb2seg.evaluation.evaluator import eval_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained FiLM-SMP model")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    eval_run_dir(Path(args.run_dir), split=args.split, num_workers=args.num_workers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
