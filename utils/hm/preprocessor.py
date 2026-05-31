"""
H&M dataset preprocessor.

Reads raw hm.csv and produces a cleaned items.csv ready for ES indexing.

Key transformation:
  mainCatCode (e.g. "ladies_cardigansjumpers_jumpers")
      → gender      : "Women"
      → item_type   : "Cardigansjumpers Jumpers"
      → gender_type : "Women Cardigansjumpers Jumpers"

Output description (used as embedding text):
  "{gender_type} | {brandName} | {colorName} | ${price} | {details}"

Usage:
    python datasets/hm/preprocessor.py
    python datasets/hm/preprocessor.py --input data/hm/raw_handm.csv --output data/hm/hm.csv
"""

import argparse
import os
import pandas as pd


# ── gender code mapping ────────────────────────────────────────────────────────
_GENDER_MAP = {
    "ladies":   "Women",
    "men":      "Men",
    "kids":     "Kids",
    "divided":  "Divided",
    "newborn":  "Newborn",
    "baby":     "Baby",
}


class HMPreprocessor:
    """Preprocess the raw H&M CSV into a standardised items.csv."""

    def __init__(self, input_path: str = "data/hm/raw_handm.csv",
                 output_path: str = "data/hm/hm.csv"):
        self.input_path  = input_path
        self.output_path = output_path

    # ── public ──────────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        print(f"Reading raw data from: {self.input_path}")
        df = pd.read_csv(self.input_path, low_memory=False)
        print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

        df = self._clean(df)
        df = self._parse_main_cat_code(df)
        df = self._deduplicate(df)
        df = self._build_description(df)
        df = self._select_output_columns(df)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved {len(df)} items to: {self.output_path}")
        return df

    # ── private ─────────────────────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnamed index column and fill missing strings."""
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate by productName.
        Same product may appear in multiple colors → merge colorName values into one row.
        Other fields are taken from the first occurrence.
        """
        before = len(df)
        # Aggregate colorName across variants; keep first for all other fields
        agg = {"colorName": lambda x: ", ".join(sorted(set(x)))}
        first_cols = [c for c in df.columns if c != "colorName"]
        for c in first_cols:
            agg[c] = "first"
        df = df.groupby("productName", sort=False, as_index=False).agg(agg)
        print(f"Deduplicated {before} → {len(df)} unique products (by productName)")
        return df

    def _parse_main_cat_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split mainCatCode into gender and item_type.
          ladies_cardigansjumpers_jumpers
              → gender    : Women
              → item_type : Cardigansjumpers Jumpers
              → gender_type: Women Cardigansjumpers Jumpers
        """
        def _parse(code: str):
            parts = code.lower().split("_")
            gender    = _GENDER_MAP.get(parts[0], parts[0].title()) if parts else ""
            item_type = " ".join(p.title() for p in parts[1:]) if len(parts) > 1 else ""
            return gender, item_type

        parsed = df["mainCatCode"].apply(_parse)
        df["gender"]      = parsed.apply(lambda x: x[0])
        df["item_type"]   = parsed.apply(lambda x: x[1])
        df["gender_type"] = df["gender"] + " " + df["item_type"]
        return df

    def _build_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine key fields into a single description string used for embedding.
        Format: "{gender_type} | {brandName} | {colorName} | ${price:.2f} | {details}"
        """
        df["description"] = (
            df["gender_type"]
            + " | " + df["brandName"]
            + " | " + df["colorName"]
            + " | $" + df["price"].apply(lambda p: f"{p:.2f}")
            + " | " + df["details"]
        )
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns needed downstream."""
        keep = [
            "productId",    # item_id
            "productName",  # title
            "brandName",
            "colorName",
            "price",
            "gender",
            "item_type",
            "gender_type",  # category (gender + type)
            "details",
            "materials",
            "description",  # embedding text
        ]
        return df[[c for c in keep if c in df.columns]]


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H&M dataset preprocessor")
    parser.add_argument("--input",  default="data/hm/hm.csv",  help="Raw CSV path")
    parser.add_argument("--output", default="data/hm/items.csv",   help="Output items CSV path")
    args = parser.parse_args()

    HMPreprocessor(input_path=args.input, output_path=args.output).run()
