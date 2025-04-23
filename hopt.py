from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import duckdb
import pandas as pd


# ── CONFIG ──────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SearchConfig:
    limit_results_star: int = 15          # max per-column DuckDB hits
    limit_res_vs: int = 25                # max per query vector hits
    sim_threshold: float = 0.15           # cosine distance threshold
    limit_spec_results: int = 10          # speculative fallback cap


# ── SMALL HELPERS ───────────────────────────────────────────────────────────────
def _val(rec: dict, field: str) -> str | None:
    """Extract and canonicalise the `field.value` string or return None."""
    v = (rec.get(field) or {}).get("value")
    return v.upper() if isinstance(v, str) else None


def _bulk_vector_search(terms: list[str],
                        *,
                        table: str,
                        limit: int,
                        threshold: float) -> list[list[dict]]:
    """
    Wrapper around your ANN/vector store so we can hit it once for many queries.
    Must return `List[List[result-dict]]` aligned with `terms`.
    Replace this stub with your own implementation.
    """
    # ---------------- YOUR ADAPTER HERE ----------------
    raise NotImplementedError("_bulk_vector_search should call your real store")


# ── MAIN ROUTINE ────────────────────────────────────────────────────────────────
def hybrid_search(companies: list[dict],
                  *,
                  cfg: SearchConfig = SearchConfig(),
                  con: duckdb.DuckDBPyConnection | None = None,
                  vector_table: str = "company_embeddings"):
    """
    Optimised rewrite of the original hybrid_search:
      • One round-trip per identifier column to DuckDB (vs. N×I)
      • One round-trip to the vector store (vs. N)
      • Post-processing done in memory
    """
    if con is None:                           # default connection if caller doesn't pass one
        con = duckdb.connect(database=":memory:")

    # ── 1)  GATHER IDENTIFIERS & CANONICAL WORDS ───────────────────────────────
    id_pool: dict[str, set[str]] = defaultdict(set)     # {'RIC': {...}, 'ISIN': {...}}
    issue_names, companies_by_issue = [], []            # preserve order for zip()

    for comp in companies:
        # keep the original 'Word' so we can annotate hits later
        for field in ("RIC", "BBticker", "Symbol", "ISIN", "SEDOL"):
            if (v := _val(comp, field)):
                id_pool[field].add(v)

        # collect issue names for bulk vector search
        issue = (comp.get("IssueName") or {}).get("value")
        if issue:
            issue_names.append(issue)
            companies_by_issue.append(comp)
        else:
            issue_names.append("")               # keep length aligned
            companies_by_issue.append(comp)

    # ── 2)  BULK QUERY DUCKDB BY IDENTIFIER ─────────────────────────────────────
    duck_hits: dict[str, pd.DataFrame] = {}
    for col, vals in id_pool.items():
        if not vals:
            continue
        placeholders = ",".join("?" for _ in vals)
        q = f"""
            SELECT *
            FROM instruments
            WHERE {col} IN ({placeholders})
            -- add LIMIT per value if desired; here we cap overall set
            LIMIT {cfg.limit_results_star * len(vals)}
        """
        duck_hits[col] = con.execute(q, list(vals)).fetch_df()

    # ── 3)  BULK VECTOR SEARCH ──────────────────────────────────────────────────
    vec_batches = _bulk_vector_search(issue_names,
                                      table=vector_table,
                                      limit=cfg.limit_res_vs,
                                      threshold=cfg.sim_threshold)

    # ── 4)  ASSEMBLE FINAL RESULTS ──────────────────────────────────────────────
    frames: list[pd.DataFrame] = []

    for comp, vec_res in zip(companies_by_issue, vec_batches):
        word = comp["Word"]

        # ---- a) direct identifier matches from DuckDB
        direct_frames = []
        for field, val in get_identifiers(comp):
            df = duck_hits.get(field)
            if df is not None:
                direct_frames.append(df[df[field] == val])

        # ---- b) speculative / vector hits -------------------------------------
        spec_rows = []
        if not direct_frames and vec_res:
            # replicate your earlier heuristic but in pure Python
            remaining = cfg.limit_spec_results
            for hit in vec_res:
                # exact matches take priority, break after first good hit
                if any(word.upper() == hit.get(col, "").upper()
                       for col in ("RIC", "BBTicker", "ISIN", "SEDOL", "Symbol")):
                    hit["Word"] = word
                    spec_rows.append(hit)
                    break
                elif remaining:
                    hit["Word"] = word
                    spec_rows.append(hit)
                    remaining -= 1
            if remaining:                 # fill with leftover hits if still room
                spec_rows.extend(vec_res[-remaining:])

        # annotate and collect
        for df in direct_frames:
            df = df.copy()
            df["Word"] = word
            frames.append(df)

        if spec_rows:
            frames.append(pd.DataFrame(spec_rows))

    # ── 5)  POST-PROCESS & RETURN ───────────────────────────────────────────────
    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["ISIN", "RIC", "BBTicker", "Symbol"], keep="first")
          .sort_values(["Word", "AverageVolume"], ascending=False)
          .drop_duplicates(subset="ISIN", keep="first")
          .reset_index(drop=True)
    )
    return out
