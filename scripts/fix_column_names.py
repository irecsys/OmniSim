"""
fix_column_names.py — Replace raw column names with human-readable labels in chat files.

Usage:
    python scripts/fix_column_names.py --config configs/system/system.yaml configs/imdb/imdb.yaml
    python scripts/fix_column_names.py --dirs chats/imdb/free chats/imdb/static chats/imdb/adaptive
"""
import os
import re
import argparse
import yaml


# Default replacements if no config is provided (imdb-specific)
# Use SHORT, natural-sounding noun phrases suitable for question templates like
# "Is {attribute} important to you?" or "Any preference on {attribute}?"
DEFAULT_REPLACEMENTS = {
    # Database column names → short natural labels
    "release_year": "release year",
    "vote_average": "rating",
    "original_language": "original language",
    "spoken_languages": "spoken languages",
    "production_companies": "production companies",
    "production_countries": "production countries",
    "imdb_rating": "rating",
    "imdb_score": "rating",
    "imdb_id": "movie ID",
    # LLM-generated underscore attribute names
    "main_characters": "main characters",
    "main_character": "main character",
    "main_characters_age": "age of main characters",
    "main_cast": "main cast",
    "main_actor": "main actor",
    "main_villain": "main villain",
    "main_theme": "main theme",
    "awards_won": "awards won",
    "plot_twist": "plot twist",
    "plot_summary": "plot summary",
    "visual_effects": "visual effects",
    "visual_style": "visual style",
    "sequel_or_series": "sequel or series",
    "sequel_or_reboot": "sequel or reboot",
    "historical_accuracy": "historical accuracy",
    "historical_era": "historical era",
    "historical_event_depicted": "historical event depicted",
    "era_setting": "era setting",
    "based_on_true_story": "based on a true story",
    "based_on": "based on",
    "ending_type": "ending type",
    "target_audience": "target audience",
    "target_age_group": "target age group",
    "publication_year": "publication year",
    "original_title": "original title",
    "number_of_episodes": "number of episodes",
    "number_of_seasons": "number of seasons",
    "critic_score": "critic score",
    "rotten_tomatoes_score": "Rotten Tomatoes score",
    "violence_level": "level of violence",
    "gore_level": "level of gore",
    "jump_scares": "jump scares",
    "horror_subgenre": "horror subgenre",
    "genre_subcategory": "genre subcategory",
    "stunt_intensity": "stunt intensity",
    "similar_movies": "similar movies",
    "scientific_concept": "scientific concept",
    "romantic_plot": "romantic plot",
    "romance_type": "romance type",
    "restoration_quality": "restoration quality",
    "production_company": "production company",
    "page_count": "page count",
    "narrative_style": "narrative style",
    "music_style": "music style",
    "music_director": "music director",
    "movie_length": "movie length",
    "location_setting": "location setting",
    "emotional_intensity": "emotional intensity",
    "dance_style": "dance style",
    "cult_status": "cult status",
    "country_of_origin": "country of origin",
    "biker_club": "biker club",
    "battle_or_war": "battle or war",
    "battle_depiction": "battle depiction",
    "faithfulness_to_source_material": "faithfulness to source material",
    "fight_choreographer": "fight choreographer",
    "episode_length": "episode length",
    "english_title": "English title",
    "target_age_range": "target age range",
    "main_character_style": "main character style",
    "fight_choreography_style": "fight choreography style",
}


def load_item_attributes(config_files):
    """Load item_attributes mapping from one or more yaml config files."""
    merged = {}
    for path in config_files:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg and "item_attributes" in cfg:
            merged.update(cfg["item_attributes"])
    return merged


def build_replacements(item_attributes: dict) -> dict:
    """
    Build replacement map: raw column key → human-readable label.
    Only include keys that won't cause cascading replacements:
    - Underscore-form keys only (e.g. release_year, not 'release year')
    - Skip keys whose label contains the key itself (e.g. 'genres'→'movie genres' would loop)
    """
    rep = {}
    all_sources = {**DEFAULT_REPLACEMENTS, **item_attributes}
    for k, v in all_sources.items():
        # Skip if label contains the key (cascading risk: replacing again would corrupt)
        if k.lower() in v.lower():
            continue
        # Skip space-variant keys (they'd re-match inside already-replaced text)
        if '_' not in k and ' ' not in k:
            # Non-underscore, single-word keys: allow only if label doesn't overlap
            rep[k] = v
        elif '_' in k:
            # Underscore keys: always safe (underscore won't appear in replacements)
            rep[k] = v
    return rep


# Cascade damage repair: fix files where replacements were applied multiple times
CASCADE_REPAIRS = [
    # Fix cascaded/long-form labels → short natural labels
    # Repeated "movie" prefixes
    (r'\b(?:movie\s+){1,}genres\b',                              "genres"),
    (r'\b(?:movie\s+){1,}production\s+companies\b',              "production companies"),
    (r'\b(?:movie\s+){1,}description\b',                         "overview"),
    # "adult only" cascade
    (r'\badult(?:\s+only)+\b',                                    "adult-only content"),
    # Long-form "of the movie" labels → short form
    (r'\boriginal\s+language(?:\s+of\s+the\s+movie)+\b',         "original language"),
    (r'\blanguages\s+spoken(?:\s+in\s+the\s+movie)+\b',          "spoken languages"),
    (r'\baverage\s+rating(?:\s+of\s+the\s+movie)+\b',            "rating"),
    (r'\byear(?:\s+of\s+the\s+movie)+\b',                        "release year"),
    (r'\bcountries\s+where\s+the\s+movie\s+was\s+produced\b',    "production countries"),
    (r'\bwas\s+produced(?:\s+was\s+produced)+\b',                "was produced"),
]


def repair_cascades(content: str) -> tuple[str, int]:
    """Fix cascade-repeated replacements."""
    count = 0
    for pattern, replacement in CASCADE_REPAIRS:
        new_content, n = re.subn(pattern, replacement, content, flags=re.IGNORECASE)
        if n:
            content = new_content
            count += n
    return content, count


def replace_in_file(filepath: str, replacements: dict, repair: bool = True) -> int:
    """Replace all occurrences in a file. Returns number of replacements made."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    original = content
    count = 0

    # First: repair any cascade damage from previous runs
    if repair:
        content, n = repair_cascades(content)
        count += n

    # Then: apply replacements (longest key first to avoid partial matches)
    for raw, label in sorted(replacements.items(), key=lambda x: -len(x[0])):
        # Word-boundary match, case-insensitive
        pattern = r'\b' + re.escape(raw) + r'\b'
        new_content, n = re.subn(pattern, label, content, flags=re.IGNORECASE)
        if n:
            content = new_content
            count += n

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    return count


def process_dirs(dirs: list, replacements: dict):
    total_files = 0
    total_replacements = 0
    for base_dir in dirs:
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith(".txt") or fname.endswith(".json"):
                    fpath = os.path.join(root, fname)
                    n = replace_in_file(fpath, replacements)
                    if n > 0:
                        total_files += 1
                        total_replacements += n
                        print(f"  [{n:3d} subs] {fpath}")
    print(f"\nDone: {total_replacements} replacements in {total_files} files.")


def main():
    parser = argparse.ArgumentParser(description="Replace raw column names in chat files")
    parser.add_argument("--config", nargs="+", help="Config YAML files to load item_attributes from")
    parser.add_argument("--dirs", nargs="+", default=["chats/imdb"], help="Directories to process")
    args = parser.parse_args()

    if args.config:
        item_attributes = load_item_attributes(args.config)
        print(f"Loaded {len(item_attributes)} attribute mappings from config.")
    else:
        item_attributes = {}

    replacements = build_replacements(item_attributes)
    print(f"Total replacement rules: {len(replacements)}")
    for k, v in sorted(replacements.items()):
        print(f"  {k!r:40s} → {v!r}")
    print()

    process_dirs(args.dirs, replacements)


if __name__ == "__main__":
    main()
