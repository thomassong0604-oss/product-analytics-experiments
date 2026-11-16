#!/usr/bin/env python3
"""Land an event stream in DuckDB, from the synthetic generator or a real public dataset.

    python scripts/seed.py --users 20000 --lift 0.12        # synthetic, known true effect
    python scripts/seed.py --source online-retail-ii --path ~/Downloads/online_retail_II.xlsx

Synthetic is the default because it is the only source whose true treatment effect is known, and
that is what the validation study needs. The real-data path exists so the warehouse models are
exercised against the mess of an actual dataset — see `pae.load` for what it does and does not
claim about the experiment in that case.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pae.generate import Spec, dataset          # noqa: E402
from pae.warehouse import DEFAULT_DB, load      # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "online-retail-ii"])
    parser.add_argument("--path", help="local file, for a real dataset")
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--days", type=int, default=56)
    parser.add_argument("--lift", type=float, default=0.12,
                        help="true relative lift on the purchase step; 0 makes it an A/A test")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    if args.source == "synthetic":
        spec = Spec(n_users=args.users, days=args.days, true_lift=args.lift, seed=args.seed)
        tables = dataset(spec)
        print(f"synthetic: {args.users:,} users, true lift {args.lift:+.1%}, seed {args.seed}")
    else:
        if not args.path:
            sys.exit("--path is required for a real dataset")
        from pae.load import online_retail_ii
        tables = online_retail_ii(args.path)
        print(f"real dataset: {args.source} from {args.path}")

    counts = load(tables, args.db)
    for name, n in counts.items():
        print(f"  raw.{name:14s} {n:>10,} rows")
    print(f"-> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
