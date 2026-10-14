import argparse
import json
import math
from dataclasses import asdict

from .prefix import prefix_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=float, nargs="+", required=True)
    parser.add_argument("--reference", type=float, nargs="+", required=True)
    parser.add_argument(
        "--family", choices=("bilateral", "unilateral"), default="bilateral"
    )
    parser.add_argument("--search", choices=("none", "count", "tempo"), default="tempo")
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--max-step", type=int, default=3)
    args = parser.parse_args()
    try:
        match = prefix_distance(
            args.query,
            args.reference,
            family=args.family,
            search=args.search,
            gap=args.gap,
            max_step=args.max_step,
        )
    except ValueError as error:
        parser.error(str(error))
    result = asdict(match)
    if not math.isfinite(match.distance):
        result["distance"] = None
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
