import argparse
import glob
import os
import sys

import openmc.data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenMC cross_sections.xml from an HDF5 data directory."
    )
    parser.add_argument(
        "--data-dir",
        default="/home/dina/ArcLink/physics_models/openmc/data/mcnp_endfb71",
        help="Directory containing *.h5 nuclear data files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for cross_sections.xml (default: <data-dir>/cross_sections.xml).",
    )
    parser.add_argument(
        "--require",
        nargs="+",
        default=["U234.h5", "U235.h5", "U238.h5", "O16.h5", "H1.h5"],
        help="Required HDF5 files that must exist before exporting XML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output = os.path.abspath(args.output) if args.output else os.path.join(data_dir, "cross_sections.xml")

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return 1

    missing = [name for name in args.require if not os.path.isfile(os.path.join(data_dir, name))]
    if missing:
        print("[ERROR] Incomplete nuclear data library. Missing required files:")
        for name in missing:
            print(f"  - {name}")
        return 2

    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[ERROR] No .h5 files found in: {data_dir}")
        return 3

    lib = openmc.data.DataLibrary()
    for h5_file in h5_files:
        lib.register_file(h5_file)
    lib.export_to_xml(output)

    print(f"[OK] Registered {len(h5_files)} files")
    print(f"[OK] Wrote cross sections XML: {output}")
    print(f"[INFO] Set OPENMC_CROSS_SECTIONS to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())