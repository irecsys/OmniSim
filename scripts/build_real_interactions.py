"""
Build real interactions.csv from IEEDATA/Dataset.npy.

Steps:
1. Load IEEDATA (userId, movieId, rating 1-10, date)
2. Keep only entries whose movieId is in items.csv (194 overlap items)
3. Randomly pick 50 of those catalog items as the "history" pool
4. Pick 15 users who have rated >= 5 of those 50 items
5. Convert rating 1-10 -> 1-5, date -> timestamp
6. Save as data/imdb/interactions.csv
7. Generate fake users.csv with demographic info

Result: interactions overlaps with at most 50 item IDs from items.csv.
Remaining ~150 catalog items can serve as unseen test targets.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime

random.seed(42)

# -- Load IEEDATA --------------------------------------------------------------
print("Loading IEEDATA...")
raw = np.load("data/imdb/IEEDATA/Dataset.npy", allow_pickle=True)
print(f"  Total rows: {len(raw):,}")

rows = []
for entry in raw:
    parts = entry.split(",", 3)
    if len(parts) == 4:
        rows.append(parts)

df = pd.DataFrame(rows, columns=["user_id", "imdb_id", "rating", "date"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])
print(f"  After parse: {len(df):,} rows, {df['user_id'].nunique():,} unique users")

# -- Keep only catalog items ---------------------------------------------------
items_df = pd.read_csv("data/imdb/items.csv")
catalog_ids = set(items_df["imdb_id"].tolist())
df = df[df["imdb_id"].isin(catalog_ids)].copy()
print(f"  After keeping only catalog items: {len(df):,} rows, "
      f"{df['imdb_id'].nunique()} unique items")

# -- Randomly pick 50 catalog items as history pool ---------------------------
all_catalog_rated = df["imdb_id"].unique().tolist()
history_items = set(random.sample(all_catalog_rated, min(50, len(all_catalog_rated))))
df = df[df["imdb_id"].isin(history_items)].copy()
print(f"  After limiting to 50 history items: {len(df):,} rows")

# -- Pick 15 users with >= 3 ratings in the history pool ----------------------
user_counts = df["user_id"].value_counts()
active_users = user_counts[user_counts >= 3].index.tolist()
print(f"  Users with >= 3 ratings in history pool: {len(active_users):,}")
chosen_users = random.sample(active_users, 15)
df = df[df["user_id"].isin(chosen_users)].copy()
print(f"  After picking 15 users: {len(df):,} rows")

# -- Remap user IDs to u1..u15 ------------------------------------------------
user_map = {uid: f"u{i+1}" for i, uid in enumerate(chosen_users)}
df["user_id"] = df["user_id"].map(user_map)

# -- Convert rating 1-10 -> 1-5 -----------------------------------------------
df["vote_average"] = df["rating"].apply(lambda r: max(1, round(r / 2)))

# -- Convert date "16 January 2005" -> "2005-01-16" ---------------------------
def parse_date(s):
    s = s.strip()
    try:
        return datetime.strptime(s, "%d %B %Y").strftime("%Y-%m-%d")
    except Exception:
        return None

df["timestamp"] = df["date"].apply(parse_date)
df = df.dropna(subset=["timestamp"])

# -- Save interactions.csv ----------------------------------------------------
out_df = df[["user_id", "imdb_id", "vote_average", "timestamp"]].sort_values(
    ["user_id", "timestamp"]
)
out_df.to_csv("data/imdb/interactions.csv", index=False)
print(f"\nSaved interactions.csv -- {len(out_df):,} rows, {out_df['user_id'].nunique()} users")
print(f"Unique items in interactions: {out_df['imdb_id'].nunique()} "
      f"(all in items.csv, overlap <= 50)")
print(out_df.head(10).to_string())

# -- Generate test_pairs.csv: each user gets 1 target from items NOT in their history --
print("\nGenerating test_pairs.csv...")
test_pairs_rows = []
for uid in [f"u{i+1}" for i in range(15)]:
    user_history = set(out_df[out_df["user_id"] == uid]["imdb_id"].tolist())
    # candidate targets = catalog items not in this user's history
    candidates = [x for x in catalog_ids if isinstance(x, str) and x not in user_history]
    if candidates:
        target = random.choice(candidates)
        test_pairs_rows.append({"user_id": uid, "item_id": target})

test_pairs_df = pd.DataFrame(test_pairs_rows)
test_pairs_df.to_csv("data/imdb/test_pairs.csv", index=False)
print(f"Saved test_pairs.csv -- {len(test_pairs_df)} pairs")
print(test_pairs_df.to_string())

# -- Generate fake users.csv --------------------------------------------------
GENDERS = ["Male", "Female"]
NATIONALITIES = [
    "United States", "United Kingdom", "Canada", "Australia",
    "Germany", "France", "India", "Brazil", "Japan", "South Korea",
]
PROFESSIONS = [
    "engineer", "teacher", "artist", "manager", "student",
    "doctor", "lawyer", "designer", "researcher", "writer",
]

users_rows = []
for uid in [f"u{i+1}" for i in range(15)]:
    users_rows.append({
        "user_id": uid,
        "birthyear": random.randint(1970, 2000),
        "gender": random.choice(GENDERS),
        "nationality": random.choice(NATIONALITIES),
        "profession": random.choice(PROFESSIONS),
    })

users_df = pd.DataFrame(users_rows)
users_df.to_csv("data/imdb/users.csv", index=False)
print(f"\nSaved users.csv -- {len(users_df)} users")
print(users_df.to_string())
