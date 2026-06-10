"""One-off: extract every result (figures + tables) into presentation/results/.

Figures: every embedded PNG output across notebooks/*.ipynb, named by the
notebook + the nearest preceding markdown heading + a running index.
Tables: every results/**/*.parquet rendered to CSV (always) and Markdown
(for the slide-friendly, not-too-wide ones). Per-ticker return panels and
daily equity curves are CSV-only (too long/wide for a slide table).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentation" / "results"
FIG_DIR = OUT / "figures"
TBL_DIR = OUT / "tables"


def df_to_markdown(df: pd.DataFrame, index: bool) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        cells = ([fmt(idx)] if index else []) + [fmt(v) for v in row]
        rows.append(cells)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def slugify(text: str, maxlen: int = 50) -> str:
    text = re.sub(r"[#*`>_]", "", text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:maxlen].strip("-") or "fig"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def extract_figures() -> list[dict]:
    manifest = []
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        stem = nb_path.stem
        nb = json.loads(nb_path.read_text())
        dest = FIG_DIR / stem
        dest.mkdir(parents=True, exist_ok=True)
        last_heading = ""
        fig_idx = 0
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown":
                for line in src.splitlines():
                    if line.lstrip().startswith("#"):
                        last_heading = line.strip()
                        break
                continue
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                png = data.get("image/png")
                if not png:
                    continue
                fig_idx += 1
                slug = slugify(last_heading) if last_heading else "fig"
                fname = f"{stem}__{fig_idx:02d}__{slug}.png"
                raw = base64.b64decode(png if isinstance(png, str) else "".join(png))
                (dest / fname).write_bytes(raw)
                manifest.append({
                    "notebook": nb_path.name,
                    "figure": f"figures/{stem}/{fname}",
                    "context_heading": last_heading.lstrip("# ").strip(),
                    "kb": round(len(raw) / 1024, 1),
                })
    return manifest


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Substrings of parquet stems that are wide/long time-series -> CSV only.
CSV_ONLY = ("_panel_", "_equity", "_distributions")


def write_tables() -> list[dict]:
    manifest = []
    sources = {
        "bist100": ROOT / "results",
        "sp500": ROOT / "results" / "sp500",
        "currency": ROOT / "results" / "currency",
    }
    for label, src_dir in sources.items():
        if not src_dir.exists():
            continue
        dest = TBL_DIR / label
        dest.mkdir(parents=True, exist_ok=True)
        for pq in sorted(src_dir.glob("*.parquet")):
            df = pd.read_parquet(pq)
            stem = pq.stem
            csv_only = any(k in stem for k in CSV_ONLY)
            # equity/panel carry a datetime index worth preserving
            df.to_csv(dest / f"{stem}.csv", index=not df.index.equals(pd.RangeIndex(len(df))))
            md_made = False
            if not csv_only:
                show = df.copy()
                keep_index = not show.index.equals(pd.RangeIndex(len(show)))
                body = df_to_markdown(show, index=keep_index)
                (dest / f"{stem}.md").write_text(f"# {label} — {stem}\n\n{body}\n")
                md_made = True
            manifest.append({
                "dataset": label,
                "table": f"tables/{label}/{stem}.csv",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "markdown": f"tables/{label}/{stem}.md" if md_made else "",
            })
    return manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = extract_figures()
    tbls = write_tables()

    lines = ["# Presentation results — extraction manifest", ""]
    lines.append(f"Auto-extracted by `scripts/extract_presentation_results.py`.")
    lines.append("")
    lines.append(f"## Figures ({len(figs)})")
    lines.append("")
    lines.append("| figure | from | context |")
    lines.append("|---|---|---|")
    for f in figs:
        lines.append(f"| `{f['figure']}` | {f['notebook']} | {f['context_heading']} |")
    lines.append("")
    lines.append(f"## Tables ({len(tbls)})")
    lines.append("")
    lines.append("| dataset | csv | rows×cols | markdown |")
    lines.append("|---|---|---|---|")
    for t in tbls:
        md = f"`{t['markdown']}`" if t["markdown"] else "_(CSV only — wide/long)_"
        lines.append(f"| {t['dataset']} | `{t['table']}` | {t['rows']}×{t['cols']} | {md} |")
    lines.append("")
    (OUT / "MANIFEST.md").write_text("\n".join(lines))
    print(f"figures: {len(figs)}  tables: {len(tbls)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
