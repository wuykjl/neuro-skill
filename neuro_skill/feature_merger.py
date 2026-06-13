"""
Feature merge — base (hand-crafted, universal) + auto (domain-specific).

Strategy: base features are SACRED. Auto features are ONLY added when
they introduce truly new domains not covered by base.
"""

from collections import Counter


def merge_features(
    base_broad: dict[str, list[str]],
    base_precise: dict[str, list[str]],
    auto_broad: dict[str, list[str]],
    auto_precise: dict[str, list[str]],
    user_broad: dict[str, list[str]] | None = None,
    user_precise: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Merge with base protection. Auto features ONLY add novel domains.

    Rules:
      1. Base features are NEVER modified
      2. Auto broad/precise features are added ONLY if their keywords
         don't match any existing base keywords (topic-level dedup)
      3. User features override everything
    """
    merged_broad = dict(base_broad)
    merged_precise = dict(base_precise)

    # Collect ALL base terms for dedup
    base_terms_all = set()
    for terms in base_broad.values():
        base_terms_all.update(t.lower() for t in terms)
    for terms in base_precise.values():
        base_terms_all.update(t.lower() for t in terms)

    # ── Quality gate for auto features ──
    # Only add auto features if they would HELP, not hurt.
    # Check: do auto features cover new skills that base misses?
    # If merging would just add noise (low-quality auto feats), skip it.

    auto_terms_all = set()
    for terms in {**auto_broad, **auto_precise}.values():
        auto_terms_all.update(t.lower() for t in terms)

    # Auto is low-quality if >30% of its terms are already in base
    overlap = auto_terms_all & base_terms_all
    unique_auto_terms = auto_terms_all - base_terms_all
    auto_quality = len(unique_auto_terms) / max(len(auto_terms_all), 1)

    if auto_quality < 0.3 or len(auto_terms_all) < 10:
        # Auto features are too noisy or too sparse — skip merge entirely
        if user_broad:
            merged_broad.update(user_broad)
        if user_precise:
            merged_precise.update(user_precise)
        return merged_broad, merged_precise

    # ── Add auto features — only novel domains ──
    for key, terms in auto_broad.items():
        if key in merged_broad:
            continue  # base already has this domain
        # Check if this is truly a new domain (different keywords)
        novel = [t for t in terms
                 if t.lower() not in base_terms_all and len(t) >= 3]
        if len(novel) >= 3:
            merged_broad[key] = novel[:10]

    for key, terms in auto_precise.items():
        if key in merged_precise:
            continue
        novel = [t for t in terms
                 if t.lower() not in base_terms_all and len(t) >= 3]
        if len(novel) >= 3:
            merged_precise[key] = novel[:8]

    # User overrides (highest priority)
    if user_broad:
        merged_broad.update(user_broad)
    if user_precise:
        merged_precise.update(user_precise)

    return merged_broad, merged_precise


def feature_quality_report(
    broad: dict[str, list[str]],
    precise: dict[str, list[str]],
    skills: list[dict],
) -> dict:
    """Quality dashboard for a feature set."""
    from neuro_skill.features import extract_skill_features

    N = len(skills)
    all_names = set(broad.keys()) | set(precise.keys())

    covered = 0
    empty = 0
    feature_counts = []
    for s in skills:
        sf = extract_skill_features(s)
        count = len(sf["broad"]) + len(sf["precise"])
        feature_counts.append(count)
        if count > 0:
            covered += 1
        else:
            empty += 1

    df = Counter()
    for s in skills:
        sf = extract_skill_features(s)
        seen = set(sf["broad"]) | set(sf["precise"])
        for f in seen:
            df[f] += 1

    high_freq = [(f, c) for f, c in df.most_common(10) if c > N * 0.8]

    from collections import defaultdict
    feat_set_to_skills = defaultdict(list)
    for s in skills:
        sf = extract_skill_features(s)
        key = frozenset(sf["broad"]) | frozenset(sf["precise"])
        feat_set_to_skills[key].append(s["name"])
    max_overlap = max(len(v) for v in feat_set_to_skills.values()) if feat_set_to_skills else 0

    def _qs():
        """Quality score (0-100). Weights calibrated on 270-skill benchmark:

        - Coverage (max 40 pts): fraction of skills with >= 1 feature.
          A healthy system covers 100% of skills.
        - Diversity (max 30 pts): feature count / target (60).
          Prevents over-reliance on too few features.
        - Overlap penalty (max 30 pts): 30 - max_skill_overlap.
          Punishes cases where many skills share identical feature sets
          (low discrimination). Typical: 5-15 skills share features.
          Severe: 50+ skills share identical features = almost zero value.
        """
        cov_s = covered / max(N, 1) * 40
        div_s = min(1.0, len(all_names) / 60) * 30
        over_s = max(0, 30 - max_overlap)
        return round(cov_s + div_s + over_s)

    return dict(
        n_skills=N, n_features=len(all_names),
        coverage=covered / max(N, 1), empty_skills=empty,
        avg_features_per_skill=sum(feature_counts) / max(N, 1),
        high_freq_features=high_freq,
        max_skill_overlap=max_overlap,
        quality_score=_qs(),
    )
