#!/usr/bin/env python3
"""Convert methanol-rdfs format RDF files to two-column plain text format.

The input format (methanol-rdfs/*.rdf) uses a structured text format with
&General, &RDF, and &Table sections. The table contains r/RDF pairs only
for bins where the RDF is non-zero, starting from the Min value specified
in the &RDF section.

The output format is a simple two-column file with all bins (NPoints rows),
where bins before the data region are filled with zeros. The r column
uses bin center positions: r_i = (i - 0.5) * step, consistent with the
project's write_rdf function in src/utils.jl.

Usage:
    python scripts/convert_methanol_rdf.py methanol-rdfs/10CH3OH-CG.rdf
    python scripts/convert_methanol_rdf.py methanol-rdfs/*.rdf
"""

import sys
from pathlib import Path


def parse_rdf_file(filepath: str) -> dict:
    """Parse a methanol-rdfs format RDF file.

    Returns a dict with keys:
        general: dict of &General section key=value pairs
        rdf_header: dict of &RDF section key=value pairs
        table: list of (r, g) tuples from the &Table section
        box_size: str (box size line after &ENDRDF)
    """
    with open(filepath) as f:
        lines = f.readlines()

    sections = {}
    current_section = None
    section_lines = []

    in_table = False
    table_data = []
    box_size = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("&End") or stripped.startswith("&END"):
            if current_section == "Table":
                in_table = False
            current_section = None
            continue

        if stripped.startswith("&") and not stripped.startswith("&End"):
            current_section = stripped.lstrip("&")
            section_lines = []
            continue

        if stripped.startswith("Box size:"):
            box_size = stripped
            continue

        if current_section == "Table":
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    r = float(parts[0])
                    g = float(parts[1])
                    table_data.append((r, g))
                except ValueError:
                    continue
        elif current_section and "=" in stripped:
            key, _, value = stripped.partition("=")
            section_lines.append((key.strip(), value.strip()))

        if current_section and current_section != "Table":
            sections[current_section] = dict(section_lines)

    sections["Table"] = table_data
    if box_size:
        sections["box_size"] = box_size

    return sections


def convert_rdf(input_path: str, output_path: str | None = None):
    """Convert a single methanol-rdfs RDF file to two-column format."""
    data = parse_rdf_file(input_path)

    general = data.get("General", {})
    rdf_header = data.get("RDF", {})
    table = data.get("Table", [])

    n_points = int(general.get("NPoints", 300))
    r_max = float(general.get("Max", 15.0))
    r_min = float(general.get("Min", 0.0))

    data_min = float(rdf_header.get("Min", 0.0))
    data_n_points = int(rdf_header.get("NPoints", len(table)))

    step = (r_max - r_min) / n_points

    if not table:
        print(f"Warning: no table data found in {input_path}", file=sys.stderr)
        return

    table_r_first = table[0][0]

    data_start_bin = int(round(data_min / step)) + 1
    n_leading_zeros = data_start_bin - 1

    if len(table) < data_n_points:
        print(
            f"Warning: table has {len(table)} rows, RDF section says NPoints={data_n_points}",
            file=sys.stderr,
        )

    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_converted.rdf")

    with open(output_path, "w") as f:
        f.write("# RDF data\n")
        f.write("# r, Å; RDF\n")
        for i in range(1, n_points + 1):
            r_center = (i - 0.5) * step
            if i <= n_leading_zeros:
                g = 0.0
            else:
                idx = i - data_start_bin
                if idx < len(table):
                    g = table[idx][1]
                else:
                    g = 0.0
            f.write(f"{r_center:.3f}    {g:.8f}\n")

    print(f"Converted {input_path} -> {output_path}")
    print(f"  Total bins: {n_points}, step: {step}")
    print(f"  Leading zeros: {n_leading_zeros}, data bins: {len(table)}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.rdf> [input2.rdf ...]", file=sys.stderr)
        sys.exit(1)

    for path in sys.argv[1:]:
        convert_rdf(path)


if __name__ == "__main__":
    main()
