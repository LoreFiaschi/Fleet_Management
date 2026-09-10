"""Create the presentation plot from one horizon-sweep report."""

from __future__ import annotations

import argparse

from fleet_management import plot_horizon_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Horizon-sweep YAML report")
    parser.add_argument("output", help="Destination PNG/PDF/SVG path")
    arguments = parser.parse_args()
    plot_horizon_sweep(arguments.report, arguments.output)
    print("plot:", arguments.output)


if __name__ == "__main__":
    main()

