"""Generate the UNION ALL query that reads every PDU table in
google.com:google-cluster-data.powerdata_2019.

The dataset has 50 cell-prefixed tables (cell{a-h}_pdu{N}), not one table
per cell, so the query has to be built rather than hand-written. Google's
own PowerData2019.md documents 57 power domains total, of which 2 belong
to cells outside the eight ClusterData2019 cells; only 50 tables are
actually present in the public BigQuery mirror under cella..cellh, a
discrepancy from the documented 55 that is recorded, not silently
resolved, in docs/official_data_alignment_audit.md.

Usage:
    python scripts/bigquery/generate_powerdata_query.py > /tmp/powerdata_all.sql
    bq query --nouse_legacy_sql --dry_run < /tmp/powerdata_all.sql   # verify cost first
    bq query --nouse_legacy_sql --format=csv --max_rows=500000 < /tmp/powerdata_all.sql \\
        > data/raw/official_google_2019/powerdata_all_pdus.csv
"""

# Discovered via `bq ls google.com:google-cluster-data:powerdata_2019` on 2026-09-19.
CELL_PDUS = {
    "a": [6, 7, 8, 9, 10],
    "b": [11, 12, 13, 14, 15],
    "c": [38, 39, 40, 41, 42, 43],
    "d": [32, 33, 34, 35, 36, 37],
    "e": [26, 27, 28, 29, 30, 31],
    "f": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "g": [1, 2, 3, 4, 5],
    "h": [44, 45, 46, 47, 48, 49, 50],
}

COLUMNS = "cell, pdu, time, measured_power_util, production_power_util, bad_measurement_data, bad_production_power_data"

def build_query() -> str:
    parts = []
    for cell, pdus in CELL_PDUS.items():
        for pdu in pdus:
            table = f"google.com:google-cluster-data.powerdata_2019.cell{cell}_pdu{pdu}"
            parts.append(f"SELECT {COLUMNS} FROM `{table}`")
    return "\nUNION ALL\n".join(parts)


if __name__ == "__main__":
    print(build_query())
