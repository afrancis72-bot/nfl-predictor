"""
PGA TOUR PREDICTOR PRO
======================
 
A modular PGA tournament prediction engine.

Core capabilities:
    - 8 / 16 / 24 / 36 round weighted recent form
    - SG Total / Approach / OTT / ARG / Putting
    - Driving distance / accuracy
    - Birdie rate / bogey avoidance
    - Par 3 / 4 / 5 performance
    - Course history
    - Course DNA / architecture
    - Approach-distance buckets
    - Weather adjustments
    - Tee-wave adjustments
    - Field-strength adjustments
    - Monte Carlo tournament simulation
    - Win / Top 5 / Top 10 / Top 20 / Make Cut
    - Expected and median finish
    - DraftKings-style scoring proxy
    - DFS value / leverage / ceiling
    - Salary-cap lineup optimization
    - Walk-forward backtesting

The program is intentionally CSV-driven so current PGA data can be
fed into it without changing the model itself.

Required packages:
    pip install pandas numpy scipy scikit-learn

Example:
    python pga_predictor_pro.py predict \
        --tournament "Example Open" \
        --sims 25000

DFS:
    python pga_predictor_pro.py lineups \
        --tournament "Example Open" \
        --sims 50000 \
        --lineups 20

Backtest:
    python pga_predictor_pro.py backtest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# OPTIONAL SCIPY MILP
# ---------------------------------------------------------------------

try:
    from scipy.optimize import milp, LinearConstraint, Bounds

    SCIPY_MILP = True
except Exception:
    SCIPY_MILP = False


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"

DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

STAT_COLS = [
    "sg_total",
    "sg_approach",
    "sg_ott",
    "sg_arg",
    "sg_putting",
    "driving_accuracy",
    "driving_distance",
    "birdie_rate",
    "bogey_avoidance",
    "par3",
    "par4",
    "par5",
]

APPROACH_BUCKETS = [
    "under_100",
    "100_125",
    "125_150",
    "150_175",
    "175_200",
    "200_plus",
]

DEFAULT_STAT_WEIGHTS = {
    "sg_total": 0.20,
    "sg_approach": 0.19,
    "sg_ott": 0.11,
    "sg_arg": 0.06,
    "sg_putting": 0.06,
    "driving_accuracy": 0.05,
    "driving_distance": 0.04,
    "birdie_rate": 0.07,
    "bogey_avoidance": 0.06,
    "par3": 0.04,
    "par4": 0.07,
    "par5": 0.05,
}


@dataclass
class Config:
    tournament: str
    sims: int = 25000
    seed: int = 42
    salary_cap: int = 50000
    roster_size: int = 6


# ---------------------------------------------------------------------
# BASIC UTILITIES
# ---------------------------------------------------------------------

def read_csv(filename: str) -> pd.DataFrame:
    # V10.5 accepts a clean single-level repository as well as the legacy
    # data/ layout. Prefer data/ when present, then fall back to repo root.
    candidates = [DATA / filename, ROOT / filename]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")

    std = x.std(ddof=0)

    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=x.index)

    return (x - x.mean()) / std


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------
# RECENT FORM
# ---------------------------------------------------------------------

def rolling_form(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate exponentially weighted form over:
        8 rounds
        16 rounds
        24 rounds
        36 rounds

    More recent rounds receive greater weight.
    """

    if results.empty:
        return pd.DataFrame()

    d = results.copy()

    d["date"] = pd.to_datetime(
        d["date"],
        errors="coerce"
    )

    d = d.sort_values(
        ["player", "date"]
    )

    output = []

    for player, group in d.groupby(
        "player",
        sort=False
    ):

        row = {
            "player": player
        }

        for window in [8, 16, 24, 36]:

            recent = group.tail(window)

            if recent.empty:
                continue

            age = np.arange(
                len(recent)
            )[::-1]

            decay = max(
                3.0,
                window / 4.0
            )

            weights = np.exp(
                -age / decay
            )

            weights /= weights.sum()

            for stat in STAT_COLS:

                if stat not in recent.columns:
                    continue

                values = pd.to_numeric(
                    recent[stat],
                    errors="coerce"
                ).to_numpy()

                valid = ~np.isnan(values)

                if not valid.any():

                    row[
                        f"{window}_{stat}"
                    ] = np.nan

                    continue

                w = weights[valid]

                w /= w.sum()

                row[
                    f"{window}_{stat}"
                ] = float(
                    np.sum(
                        values[valid] * w
                    )
                )

        output.append(row)

    return pd.DataFrame(output)


# ---------------------------------------------------------------------
# CURRENT PLAYER FORM
# ---------------------------------------------------------------------

def current_players(
    players: pd.DataFrame,
    results: pd.DataFrame
) -> pd.DataFrame:

    p = players.copy()

    form = rolling_form(results)

    if not form.empty:

        p = p.merge(
            form,
            on="player",
            how="left"
        )

    # Combine the four windows.
    for stat in STAT_COLS:

        candidates = []

        for window, weight in [
            (8, 0.35),
            (16, 0.30),
            (24, 0.20),
            (36, 0.15),
        ]:

            column = f"{window}_{stat}"

            if column in p.columns:

                candidates.append(
                    (
                        p[column],
                        weight
                    )
                )

        if not candidates:
            continue

        matrix = np.column_stack(
            [
                series.to_numpy(float)
                for series, _ in candidates
            ]
        )

        weights = np.array(
            [
                weight
                for _, weight in candidates
            ]
        )

        valid = ~np.isnan(matrix)

        numerator = np.nansum(
            matrix * weights,
            axis=1
        )

        denominator = np.sum(
            valid * weights,
            axis=1
        )

        result = np.divide(
            numerator,
            denominator,
            out=np.full(
                len(p),
                np.nan
            ),
            where=denominator > 0
        )

        p[f"form_{stat}"] = result

    return p


# ---------------------------------------------------------------------
# FIELD STRENGTH
# ---------------------------------------------------------------------

def field_strength_adjust(
    players: pd.DataFrame,
    results: pd.DataFrame
) -> pd.DataFrame:

    p = players.copy()

    if (
        not results.empty
        and "field_strength" in results.columns
    ):

        historical = (
            results
            .groupby("player")[
                "field_strength"
            ]
            .mean()
            .rename(
                "historical_field_strength"
            )
        )

        p = p.merge(
            historical,
            on="player",
            how="left"
        )

    else:

        p["historical_field_strength"] = np.nan

    if "field_strength" not in p.columns:

        p["field_strength"] = (
            p["historical_field_strength"]
        )

    p["field_strength"] = pd.to_numeric(
        p["field_strength"],
        errors="coerce"
    )

    return p


# ---------------------------------------------------------------------
# COURSE HISTORY
# ---------------------------------------------------------------------

def course_history_score(
    history: pd.DataFrame,
    tournament: str
) -> pd.DataFrame:

    if history.empty:
        return pd.DataFrame(
            columns=[
                "player",
                "course_history_score"
            ]
        )

    if "player" not in history.columns:
        return pd.DataFrame(
            columns=[
                "player",
                "course_history_score"
            ]
        )

    d = history.copy()

    if "tournament" in d.columns:

        d = d[
            d["tournament"]
            .astype(str)
            .str.lower()
            ==
            tournament.lower()
        ]

    if d.empty:
        return pd.DataFrame(
            columns=[
                "player",
                "course_history_score"
            ]
        )

    rows = []

    for player, group in d.groupby(
        "player"
    ):

        if "course_sg_total" in group.columns:

            raw = pd.to_numeric(
                group["course_sg_total"],
                errors="coerce"
            ).mean()

        elif "finish" in group.columns:

            finish = pd.to_numeric(
                group["finish"],
                errors="coerce"
            )

            raw = -finish.mean() / 10.0

        else:

            raw = 0.0

        sample_size = len(group)

        reliability = (
            sample_size
            /
            (sample_size + 4.0)
        )

        rows.append(
            [
                player,
                raw * reliability
            ]
        )

    output = pd.DataFrame(
        rows,
        columns=[
            "player",
            "course_history_score"
        ]
    )

    output[
        "course_history_score"
    ] = zscore(
        output[
            "course_history_score"
        ]
    )

    return output


# ---------------------------------------------------------------------
# COURSE DNA
# ---------------------------------------------------------------------

def course_dna(
    holes: pd.DataFrame,
    tournament: str
) -> Dict[str, float]:

    # V10.9: researched, pre-event Course DNA for the 2026 Baycurrent Classic.
    # This profile is intentionally based on course/setup evidence rather than
    # fitting to 2026 outcomes.  Weather remains a separate round adjustment.
    tkey = str(tournament or "").casefold()
    if "baycurrent" in tkey:
        return {
            "length": 0.62,
            "narrow": 0.38,
            "rough": 0.42,
            "water": 0.28,
            "bunker": 0.72,
            "green_small": 0.60,
            "wind_exposure": 0.50,
            "elevation": 0.35,
            "par3": 3.0 / 18.0,
            "par4": 13.0 / 18.0,
            "par5": 2.0 / 18.0,
            "research_profile": "baycurrent_2026",
        }

    if holes.empty:

        return {
            "length": 0.50,
            "narrow": 0.50,
            "rough": 0.50,
            "water": 0.50,
            "bunker": 0.50,
            "green_small": 0.50,
            "wind_exposure": 0.50,
            "elevation": 0.50,
            "par3": 0.25,
            "par4": 0.50,
            "par5": 0.25,
        }

    d = holes.copy()

    if "tournament" in d.columns:

        d = d[
            d["tournament"]
            .astype(str)
            .str.lower()
            ==
            tournament.lower()
        ]

    if d.empty:

        return course_dna(
            pd.DataFrame(),
            tournament
        )

    def average(
        column,
        default=0.5
    ):

        if column not in d.columns:
            return default

        value = pd.to_numeric(
            d[column],
            errors="coerce"
        ).mean()

        if pd.isna(value):
            return default

        return float(value)

    yardage = average(
        "yardage",
        450
    )

    fairway_width = average(
        "fairway_width",
        30
    )

    rough = average(
        "rough_severity"
    )

    water = average(
        "water"
    )

    bunker = average(
        "bunker_density"
    )

    green_size = average(
        "green_size",
        6000
    )

    wind = average(
        "wind_exposure"
    )

    elevation = average(
        "elevation_change"
    )

    pars = d.groupby("par").size()

    total = max(
        pars.sum(),
        1
    )

    return {

        "length":
            float(
                clip01(
                    (yardage - 380)
                    / 130
                )
            ),

        "narrow":
            float(
                clip01(
                    1 -
                    (fairway_width - 20)
                    / 30
                )
            ),

        "rough":
            float(
                clip01(rough)
            ),

        "water":
            float(
                clip01(water)
            ),

        "bunker":
            float(
                clip01(bunker)
            ),

        "green_small":
            float(
                clip01(
                    1 -
                    (green_size - 4500)
                    / 4500
                )
            ),

        "wind_exposure":
            float(
                clip01(wind)
            ),

        "elevation":
            float(
                clip01(
                    elevation / 50
                )
            ),

        "par3":
            float(
                pars.get(3, 0)
                / total
            ),

        "par4":
            float(
                pars.get(4, 0)
                / total
            ),

        "par5":
            float(
                pars.get(5, 0)
                / total
            ),
    }


# ---------------------------------------------------------------------
# COURSE-SPECIFIC STAT WEIGHTS
# ---------------------------------------------------------------------

def course_stat_weights(
    dna: Dict[str, float]
) -> Dict[str, float]:

    # Independent 2026 Yokohama thesis: approach/GIR creation and OTT quality
    # are the dominant pillars; 13 par 4s elevate par-4 scoring and bogey
    # avoidance. ARG and bentgrass putting matter, but remain secondary.
    if dna.get("research_profile") == "baycurrent_2026":
        return {
            "sg_total": 0.14,
            "sg_approach": 0.22,
            "sg_ott": 0.15,
            "sg_arg": 0.05,
            "sg_putting": 0.05,
            "driving_accuracy": 0.05,
            "driving_distance": 0.08,
            "birdie_rate": 0.05,
            "bogey_avoidance": 0.08,
            "par3": 0.02,
            "par4": 0.09,
            "par5": 0.02,
        }

    weights = (
        DEFAULT_STAT_WEIGHTS.copy()
    )

    weights["sg_ott"] *= (
        1 +
        0.65 * dna["length"]
    )

    weights["driving_distance"] *= (
        1 +
        0.90 * dna["length"]
    )

    weights["driving_accuracy"] *= (
        1 +
        0.70 * dna["narrow"]
    )

    weights["sg_approach"] *= (
        1 +
        0.45 * dna["rough"]
    )

    weights["bogey_avoidance"] *= (
        1 +
        0.60 *
        max(
            dna["rough"],
            dna["water"]
        )
    )

    weights["sg_arg"] *= (
        1 +
        0.45 * dna["green_small"]
    )

    weights["sg_putting"] *= (
        1 +
        0.25 * dna["green_small"]
    )

    weights["par3"] *= (
        1 +
        2 * dna["par3"]
    )

    weights["par4"] *= (
        1 +
        1.2 * dna["par4"]
    )

    weights["par5"] *= (
        1 +
        1.4 * dna["par5"]
    )

    total = sum(
        weights.values()
    )

    return {
        key: value / total
        for key, value
        in weights.items()
    }


# ---------------------------------------------------------------------
# APPROACH DISTANCE FIT
# ---------------------------------------------------------------------

def approach_fit(
    player_row: pd.Series,
    holes: pd.DataFrame,
    tournament: str
) -> float:

    if (
        holes.empty
        or
        "approach_bucket"
        not in holes.columns
    ):
        return 0.0

    h = holes.copy()

    if "tournament" in h.columns:

        h = h[
            h["tournament"]
            .astype(str)
            .str.lower()
            ==
            tournament.lower()
        ]

    if h.empty:
        return 0.0

    counts = (
        h["approach_bucket"]
        .value_counts(
            normalize=True
        )
    )

    score = 0.0
    used = 0.0

    for bucket in APPROACH_BUCKETS:

        column = (
            f"sg_app_{bucket}"
        )

        if (
            column in player_row
            and
            bucket in counts
            and
            pd.notna(
                player_row[column]
            )
        ):

            score += (
                float(
                    player_row[column]
                )
                *
                float(
                    counts[bucket]
                )
            )

            used += float(
                counts[bucket]
            )

    if used == 0:
        return 0.0

    return score / used


# ---------------------------------------------------------------------
# WEATHER
# ---------------------------------------------------------------------

def weather_effect(
    player: pd.Series,
    weather: pd.DataFrame,
    tournament: str
) -> float:

    if weather.empty:
        return 0.0

    w = weather.copy()

    if "tournament" in w.columns:

        w = w[
            w["tournament"]
            .astype(str)
            .str.lower()
            ==
            tournament.lower()
        ]

    if w.empty:
        return 0.0

    if "player" in w.columns:

        player_weather = w[
            w["player"]
            .astype(str)
            ==
            str(player["player"])
        ]

        if player_weather.empty:
            player_weather = w

    else:

        player_weather = w

    def average(
        column,
        default
    ):

        if column not in player_weather.columns:
            return default

        value = pd.to_numeric(
            player_weather[column],
            errors="coerce"
        ).mean()

        if pd.isna(value):
            return default

        return float(value)

    wind = average(
        "wind_mph",
        10
    )

    rain = average(
        "rain_prob",
        20
    )

    temperature = average(
        "temperature_f",
        72
    )

    adjustment = 0.0

    adjustment -= (
        0.018 *
        max(
            wind - 10,
            0
        )
    )

    adjustment -= (
        0.0015 *
        max(
            rain - 20,
            0
        )
    )

    adjustment -= (
        0.002 *
        abs(
            temperature - 72
        )
    )

    # Player-specific wind skill.
    if (
        "wind_sg" in player
        and
        pd.notna(
            player["wind_sg"]
        )
    ):

        adjustment += (
            float(
                player["wind_sg"]
            )
            *
            max(
                wind - 8,
                0
            )
            / 12
        )

    # Player-specific rain skill.
    if (
        "rain_sg" in player
        and
        pd.notna(
            player["rain_sg"]
        )
    ):

        adjustment += (
            float(
                player["rain_sg"]
            )
            *
            rain
            / 100
        )

    return float(
        adjustment
    )


# ---------------------------------------------------------------------
# TEE WAVE
# ---------------------------------------------------------------------

def tee_wave_effect(
    player: pd.Series,
    weather: pd.DataFrame,
    tournament: str
) -> float:

    if (
        weather.empty
        or
        "tee_wave" not in weather.columns
    ):
        return 0.0

    w = weather.copy()

    if "tournament" in w.columns:

        w = w[
            w["tournament"]
            .astype(str)
            .str.lower()
            ==
            tournament.lower()
        ]

    if w.empty:
        return 0.0

    if "player" in w.columns:

        player_weather = w[
            w["player"]
            .astype(str)
            ==
            str(player["player"])
        ]

        if player_weather.empty:
            player_weather = w

    else:

        player_weather = w

    if (
        "wind_mph"
        not in player_weather.columns
    ):
        return 0.0

    if (
        "wind_mph"
        not in w.columns
    ):
        return 0.0

    player_wind = pd.to_numeric(
        player_weather["wind_mph"],
        errors="coerce"
    ).mean()

    field_wind = pd.to_numeric(
        w["wind_mph"],
        errors="coerce"
    ).mean()

    if (
        pd.isna(player_wind)
        or
        pd.isna(field_wind)
    ):
        return 0.0

    return float(
        0.025 *
        (
            field_wind -
            player_wind
        )
    )


# ---------------------------------------------------------------------
# BUILD MODEL FEATURES
# ---------------------------------------------------------------------

def build_features(
    players,
    results,
    history,
    holes,
    weather,
    tournament
):

    p = current_players(
        players,
        results
    )

    p = field_strength_adjust(
        p,
        results
    )

    history_score = (
        course_history_score(
            history,
            tournament
        )
    )

    p = p.merge(
        history_score,
        on="player",
        how="left"
    )

    p[
        "course_history_score"
    ] = p[
        "course_history_score"
    ].fillna(0)

    dna = course_dna(
        holes,
        tournament
    )

    weights = course_stat_weights(
        dna
    )

    score = np.zeros(
        len(p)
    )

    for stat, weight in weights.items():

        column = (
            f"form_{stat}"
        )

        if column in p.columns:

            base = p[column]

        elif stat in p.columns:

            base = p[stat]

        else:

            base = pd.Series(
                np.nan,
                index=p.index
            )

        score += (
            weight
            *
            zscore(base)
            .fillna(0)
            .to_numpy()
        )

    # Course-specific emphasis.
    if "form_sg_ott" in p:

        score += (
            0.08
            *
            zscore(
                p["form_sg_ott"]
            )
            .fillna(0)
            .to_numpy()
            *
            dna["length"]
        )

    if (
        "form_driving_accuracy"
        in p
    ):

        score += (
            0.06
            *
            zscore(
                p[
                    "form_driving_accuracy"
                ]
            )
            .fillna(0)
            .to_numpy()
            *
            dna["narrow"]
        )

    if (
        "form_sg_approach"
        in p
    ):

        score += (
            0.08
            *
            zscore(
                p[
                    "form_sg_approach"
                ]
            )
            .fillna(0)
            .to_numpy()
            *
            dna["rough"]
        )

    # Approach bucket fit.
    p["approach_fit"] = [
        approach_fit(
            row,
            holes,
            tournament
        )
        for _, row in p.iterrows()
    ]

    score += (
        0.08
        *
        zscore(
            p["approach_fit"]
        )
        .fillna(0)
        .to_numpy()
    )

    # Course history.
    score += (
        0.08
        *
        np.clip(
            p[
                "course_history_score"
            ].to_numpy(float),
            -2.5,
            2.5
        )
    )

    # Field strength.
    if "field_strength" in p:

        score += (
            0.04
            *
            zscore(
                p["field_strength"]
            )
            .fillna(0)
            .to_numpy()
        )

    # Weather.
    p["weather_adj"] = [
        weather_effect(
            row,
            weather,
            tournament
        )
        for _, row in p.iterrows()
    ]

    p["tee_wave_adj"] = [
        tee_wave_effect(
            row,
            weather,
            tournament
        )
        for _, row in p.iterrows()
    ]

    score += (
        p["weather_adj"]
        .to_numpy()
    )

    score += (
        p["tee_wave_adj"]
        .to_numpy()
    )
    # -------------------------------------------------------------
    # OWGR BASELINE STRENGTH
    # -------------------------------------------------------------
    # Version 0 baseline for historical testing.
    # Lower OWGR = stronger golfer.
    #
    # We use log rank because the difference between World #1
    # and World #10 should matter considerably more than the
    # difference between World #401 and World #410.

    if "owgr" in p.columns:

        owgr = pd.to_numeric(
            p["owgr"],
            errors="coerce"
        )

        # Give unranked players a conservative fallback ranking.
        fallback_rank = max(
            500,
            int(owgr.max()) + 50
            if owgr.notna().any()
            else 500
        )

        owgr = owgr.fillna(
            fallback_rank
        ).clip(lower=1)

        # Convert ranking into player strength.
        # #1 receives the highest raw strength.
        owgr_strength = -np.log(
            owgr
        )

        owgr_z = zscore(
            pd.Series(
                owgr_strength,
                index=p.index
            )
        ).fillna(0)

        # For this initial baseline test, OWGR carries substantial
        # weight because the other data files are not populated yet.
        score += (
            1.00 *
            owgr_z.to_numpy()
        )

        p["owgr_used"] = owgr
        p["owgr_strength"] = owgr_z
    # V10.6.6: weekly OTIS Advanced Course-Fit player layer.
    # OTIS Rank and OTIS Model are deliberately excluded. We use only the
    # decomposed within-field signals: True Skill, Course Fit and Form.
    # The scores are percentile ranks for THIS field, so z-scoring here keeps
    # them relative and avoids pretending they are strokes gained.
    if "otis_true_skill" in p.columns:
        true_skill = pd.to_numeric(p["otis_true_skill"], errors="coerce")
        form = pd.to_numeric(p.get("otis_form", pd.Series(np.nan, index=p.index)), errors="coerce")
        generic_fit = pd.to_numeric(p.get("otis_course_fit", pd.Series(np.nan, index=p.index)), errors="coerce")
        # V10.9.1: for Baycurrent/Yokohama, replace the generic weekly Course Fit
        # with the independent researched DNA signal using only components that
        # are actually present.  22 APP / 15 OTT / 5 ARG / 5 PUTT are
        # renormalized across available inputs; no missing course traits are invented.
        if dna.get("research_profile") == "baycurrent_2026":
            dna_specs = [("otis_fit_app",22.0),("otis_fit_ott",15.0),("otis_fit_arg",5.0),("otis_fit_putt",5.0)]
            dna_num = pd.Series(0.0, index=p.index)
            dna_den = pd.Series(0.0, index=p.index)
            for dna_col, dna_w in dna_specs:
                if dna_col not in p.columns:
                    continue
                dna_x = pd.to_numeric(p[dna_col], errors="coerce")
                dna_ok = dna_x.notna()
                dna_num.loc[dna_ok] += dna_w * dna_x.loc[dna_ok]
                dna_den.loc[dna_ok] += dna_w
            fit = (dna_num / dna_den.replace(0, np.nan)).fillna(generic_fit)
            p["course_fit_source"] = "Independent Yokohama DNA"
        else:
            fit = generic_fit
            p["course_fit_source"] = "OTIS Course Fit"
        p["independent_course_fit"] = fit
        form_rds = pd.to_numeric(p.get("otis_form_rds", pd.Series(np.nan, index=p.index)), errors="coerce")
        venue_rds = pd.to_numeric(p.get("otis_venue_rds", pd.Series(np.nan, index=p.index)), errors="coerce")

        # Confidence shrinkage is applied to the two more sample-sensitive layers.
        # True Skill already includes OTIS's own thin-sample shrinkage.
        form_rel = (form_rds / (form_rds + 24.0)).fillna(0.50).clip(0, 1)
        venue_rel = (venue_rds / (venue_rds + 8.0)).fillna(0.0).clip(0, 1)
        form_center = form.median(skipna=True) if form.notna().any() else 50.0
        fit_center = fit.median(skipna=True) if fit.notna().any() else 50.0
        form_adj = form_center + form_rel * (form - form_center)

        # Course Fit already contains APP/OTT/ARG/PUTT/History. Reduce its
        # confidence modestly when venue history is absent without erasing the
        # non-history components (80% of OTIS's published fit construction).
        fit_conf = 0.80 + 0.20 * venue_rel
        fit_adj = fit_center + fit_conf * (fit - fit_center)

        p["otis_form_adjusted"] = form_adj
        p["otis_course_fit_adjusted"] = fit_adj
        p["otis_form_reliability"] = form_rel
        p["otis_venue_reliability"] = venue_rel

        # Baseline player quality is the anchor. Form and course fit are bounded
        # tilts; this avoids recreating OTIS's 45/35/20 Model score.
        score += 1.00 * zscore(true_skill).fillna(0).to_numpy()
        score += 0.25 * zscore(form_adj).fillna(0).to_numpy()
        score += 0.20 * zscore(fit_adj).fillna(0).to_numpy()

    # Final model strength.
    p["model_strength"] = score

    p["strength_z"] = (
        zscore(
            p["model_strength"]
        )
        .fillna(0)
    )

    # Lower expected round score = better.
    p["expected_round"] = (
        -0.58 *
        p["strength_z"]
    )

    if "scoring_sd" in p.columns:

        scoring_sd = pd.to_numeric(
            p["scoring_sd"],
            errors="coerce"
        )

    else:

        scoring_sd = pd.Series(
            2.75,
            index=p.index
        )

    scoring_sd = (
        scoring_sd
        .fillna(2.75)
    )

    p["round_sd"] = np.clip(
        scoring_sd
        -
        0.10 *
        p["strength_z"],
        2.0,
        3.7
    )

    # Ceiling indicator.
    birdie = pd.to_numeric(
        p.get(
            "birdie_rate",
            pd.Series(
                np.nan,
                index=p.index
            )
        ),
        errors="coerce"
    )

    # V10.1: course-fit ceiling. This supplements overall strength with the
    # skills that create tournament-winning upside, while retaining putting risk.
    recent_app = pd.to_numeric(p.get("form_sg_approach", p.get("sg_approach", pd.Series(np.nan, index=p.index))), errors="coerce")
    season_app = pd.to_numeric(p.get("sg_approach", pd.Series(np.nan, index=p.index)), errors="coerce")
    putting = pd.to_numeric(p.get("sg_putting", pd.Series(np.nan, index=p.index)), errors="coerce")
    approach_course_weight = 0.30 + 0.10 * (1.0 - dna.get("green_small", 0.5))
    p["course_fit_ceiling"] = (
        approach_course_weight * zscore(season_app).fillna(0)
        + 0.25 * zscore(recent_app).fillna(0)
        + 0.20 * zscore(birdie).fillna(0)
        + 0.10 * zscore(p["approach_fit"]).fillna(0)
        + 0.10 * zscore(putting).fillna(0)
        + 0.05 * np.clip(p["course_history_score"].to_numpy(float), -2.5, 2.5)
    )
    p["ceiling_score"] = (
        p["strength_z"]
        + 0.30 * p["course_fit_ceiling"]
        + 0.30 * zscore(birdie).fillna(0)
    )

    return (
        p,
        dna,
        weights
    )


# ---------------------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------------------

def simulate(
    p: pd.DataFrame,
    sims: int = 25000,
    seed: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(
        seed
    )

    n = len(p)

    if n == 0:
        return pd.DataFrame()

    mu = p[
        "expected_round"
    ].to_numpy(float)

    sd = p[
        "round_sd"
    ].to_numpy(float)

    # V10.2: Course-Fit Ceiling is applied upstream, inside the tournament
    # distribution rather than as an optimizer-only bonus. Positive fit lowers
    # expected strokes modestly and receives a very small upside-dispersion bump.
    # Both effects are bounded to prevent a course-fit score from dominating the
    # golfer's underlying ability.
    if "course_fit_ceiling" in p.columns:
        cf = pd.to_numeric(p["course_fit_ceiling"], errors="coerce").fillna(0.0)
        cf_sd = cf.std(ddof=0)
        cf_z = ((cf - cf.mean()) / cf_sd).fillna(0.0) if cf_sd and not np.isnan(cf_sd) else pd.Series(0.0, index=p.index)
        coverage = pd.to_numeric(p.get("course_fit_coverage", pd.Series(1.0, index=p.index)), errors="coerce").fillna(0.0).clip(0, 1)
        fit_signal = cf_z.to_numpy(float) * coverage.to_numpy(float)
        mu = mu + np.clip(-0.22 * fit_signal, -0.45, 0.45)
        sd = sd * np.clip(1.0 + 0.025 * np.maximum(fit_signal, 0.0), 1.0, 1.06)

    # Four-round tournament simulation.
    course_shock = rng.normal(
        0,
        0.35,
        size=(
            sims,
            1,
            4
        )
    )

    player_latent = rng.normal(
        0,
        1,
        size=(
            sims,
            n,
            1
        )
    )

    round_noise = rng.normal(
        0,
        1,
        size=(
            sims,
            n,
            4
        )
    )

    rounds = (
        mu[None, :, None]
        +
        0.45 *
        sd[None, :, None]
        *
        player_latent
        +
        0.89 *
        sd[None, :, None]
        *
        round_noise
        +
        course_shock
    )

    # Cut after 36 holes.
    after_two = (
        rounds[:, :, :2]
        .sum(axis=2)
    )

    cut_size = max(
        1,
        n // 2
    )

    cutoff = np.partition(
        after_two,
        cut_size - 1,
        axis=1
    )[:, cut_size - 1]

    made_cut = (
        after_two
        <=
        cutoff[:, None]
    )

    final_score = (
        rounds.sum(axis=2)
    )

    # Players missing the cut receive a very large
    # effective score so they cannot rank above made-cut players.
    scored = np.where(
        made_cut,
        final_score,
        final_score + 50
    )

    tie_break = rng.normal(
        0,
        1e-6,
        scored.shape
    )

    order = np.argsort(
        scored + tie_break,
        axis=1
    )

    ranks = np.empty_like(
        order
    )

    ranks[
        np.arange(sims)[:, None],
        order
    ] = np.arange(
        1,
        n + 1
    )[None, :]

    output = pd.DataFrame({

        "player":
            p["player"],

        "win_pct":
            (ranks == 1)
            .mean(axis=0),

        "top5_pct":
            (ranks <= 5)
            .mean(axis=0),

        "top10_pct":
            (ranks <= 10)
            .mean(axis=0),

        "top20_pct":
            (ranks <= 20)
            .mean(axis=0),

        "make_cut_pct":
            made_cut.mean(axis=0),

        "expected_finish":
            ranks.mean(axis=0),

        "median_finish":
            np.median(
                ranks,
                axis=0
            ),

        "expected_4round_score":
            final_score.mean(axis=0),

        "score_sd":
            final_score.std(axis=0),
    })

    # DraftKings-style proxy.
    output["dk_proxy"] = (
        8 *
        output["make_cut_pct"]
        +
        4 *
        (ranks <= 30).mean(axis=0)
        +
        6 *
        output["top20_pct"]
        +
        8 *
        output["top10_pct"]
        +
        8 *
        output["top5_pct"]
        +
        10 *
        output["win_pct"]
    )

    return output


# ---------------------------------------------------------------------
# PREDICT TOURNAMENT
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# PREDICT FROM CURRENT-WEEK DATAFRAMES (V10.5)
# ---------------------------------------------------------------------

def predict_from_dataframes(
    config: Config,
    players: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    holes: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the existing tournament model directly from weekly inputs.

    This is an ingestion wrapper only; it does not change model weights,
    simulation math, or optimizer behavior.
    """
    if players is None or players.empty:
        raise RuntimeError("Current-week players/DK field is missing or empty.")

    p = players.copy()
    for frame in [player_stats]:
        if frame is not None and not frame.empty:
            if "player" not in frame.columns:
                raise RuntimeError("player_stats.csv must contain a 'player' column.")
            # Avoid duplicate salary/identity columns from weekly joins.
            add_cols = [c for c in frame.columns if c == "player" or c not in p.columns]
            p = p.merge(frame[add_cols], on="player", how="left")

    results = pd.DataFrame() if results is None else results.copy()
    history = pd.DataFrame() if history is None else history.copy()
    holes = pd.DataFrame() if holes is None else holes.copy()
    weather = pd.DataFrame() if weather is None else weather.copy()

    features, dna, weights = build_features(
        p, results, history, holes, weather, config.tournament
    )
    simulation = simulate(features, sims=config.sims, seed=config.seed)
    final = features.merge(simulation, on="player", how="left")

    if "salary" in final.columns:
        final["salary"] = pd.to_numeric(final["salary"], errors="coerce")
        final["points_per_1k"] = final["dk_proxy"] / (final["salary"] / 1000)
    if "ownership" in final.columns:
        final["ownership"] = pd.to_numeric(final["ownership"], errors="coerce")
        final["leverage"] = final["dk_proxy"] / np.maximum(final["ownership"], 0.25)
    else:
        final["ownership"] = np.nan
        final["leverage"] = np.nan

    final["model_rank"] = final["expected_finish"].rank(method="min")
    final = final.sort_values(["win_pct", "top10_pct", "expected_finish"], ascending=[False, False, True])
    return final

def predict(
    config: Config
):

    players = read_csv(
        "players.csv"
    )

    # Optional pre-tournament player statistics.
    player_stats = read_csv(
        "player_stats.csv"
    )

    if not player_stats.empty:
        players = players.merge(
            player_stats,
            on="player",
            how="left"
        )

    results = read_csv(
        "results.csv"
    )

    history = read_csv(
        "course_history.csv"
    )

    holes = read_csv(
        "course_holes.csv"
    )

    weather = read_csv(
        "weather.csv"
    )

    if players.empty:

        raise RuntimeError(
            "data/players.csv is missing or empty."
        )

    features, dna, weights = (
        build_features(
            players,
            results,
            history,
            holes,
            weather,
            config.tournament
        )
    )

    simulation = simulate(
        features,
        sims=config.sims,
        seed=config.seed
    )

    final = features.merge(
        simulation,
        on="player",
        how="left"
    )

    if "salary" in final.columns:

        final["salary"] = pd.to_numeric(
            final["salary"],
            errors="coerce"
        )

        final["points_per_1k"] = (
            final["dk_proxy"]
            /
            (final["salary"] / 1000)
        )

    if "ownership" in final.columns:

        final["ownership"] = pd.to_numeric(
            final["ownership"],
            errors="coerce"
        )

        final["leverage"] = (
            final["dk_proxy"]
            /
            np.maximum(
                final["ownership"],
                0.25
            )
        )

    else:

        final["ownership"] = np.nan
        final["leverage"] = np.nan

    final["model_rank"] = (
        final["expected_finish"]
        .rank(
            method="min"
        )
    )

    final = final.sort_values(
        [
            "win_pct",
            "top10_pct",
            "expected_finish"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    safe_name = (
        config.tournament
        .replace(" ", "_")
        .replace("/", "_")
    )

    final.to_csv(
        OUT /
        f"{safe_name}_predictions.csv",
        index=False
    )

    pd.DataFrame(
        [dna]
    ).to_csv(
        OUT /
        f"{safe_name}_course_dna.csv",
        index=False
    )

    pd.DataFrame(
        [weights]
    ).to_csv(
        OUT /
        f"{safe_name}_weights.csv",
        index=False
    )

    return final


# ---------------------------------------------------------------------
# DFS LINEUP OPTIMIZER
# ---------------------------------------------------------------------

def optimize_lineups(
    prediction: pd.DataFrame,
    salary_cap: int = 50000,
    roster_size: int = 6,
    count: int = 20,
    seed: int = 42
) -> pd.DataFrame:

    if "salary" not in prediction.columns:

        raise RuntimeError(
            "players.csv must contain salary."
        )

    d = prediction.dropna(
        subset=[
            "salary",
            "dk_proxy"
        ]
    ).copy()

    if len(d) < roster_size:

        raise RuntimeError(
            "Not enough players for lineup construction."
        )

    d["salary"] = (
        d["salary"]
        .astype(int)
    )

    d["dk_proxy"] = (
        d["dk_proxy"]
        .astype(float)
    )

    rng = np.random.default_rng(
        seed
    )

    lineups = []

    for iteration in range(count):

        ownership = (
            d["ownership"]
            .fillna(
                d["ownership"]
                .median()
                if d["ownership"]
                .notna()
                .any()
                else 10
            )
            .clip(lower=0.25)
        )

        ceiling = (
            d["dk_proxy"]
            +
            3 *
            d["top5_pct"]
            +
            5 *
            d["win_pct"]
        )

        leverage_bonus = (
            1 /
            np.sqrt(
                ownership
            )
        )

        objective = (
            ceiling.to_numpy()
            +
            2.5 *
            leverage_bonus.to_numpy()
            +
            rng.normal(
                0,
                0.8,
                len(d)
            )
        )

        # Exact MILP optimizer if SciPy supports it.
        if SCIPY_MILP:

            c = -objective

            integrality = np.ones(
                len(d)
            )

            constraint_matrix = (
                np.vstack(
                    [
                        d[
                            "salary"
                        ].to_numpy(),
                        np.ones(
                            len(d)
                        )
                    ]
                )
            )

            constraints = (
                LinearConstraint(
                    constraint_matrix,
                    [
                        0,
                        roster_size
                    ],
                    [
                        salary_cap,
                        roster_size
                    ]
                )
            )

            result = milp(
                c=c,
                integrality=integrality,
                bounds=Bounds(
                    np.zeros(
                        len(d)
                    ),
                    np.ones(
                        len(d)
                    )
                ),
                constraints=constraints
            )

            if not result.success:
                continue

            chosen = np.where(
                result.x > 0.5
            )[0]

        else:

            # Fallback greedy optimizer.
            order = np.argsort(
                objective
            )[::-1]

            chosen = []
            salary_used = 0

            for idx in order:

                if (
                    len(chosen)
                    >=
                    roster_size
                ):
                    break

                salary = int(
                    d.iloc[idx]
                    ["salary"]
                )

                if (
                    salary_used
                    +
                    salary
                    <=
                    salary_cap
                ):

                    chosen.append(
                        idx
                    )

                    salary_used += salary

            chosen = np.array(
                chosen
            )

        if (
            len(chosen)
            !=
            roster_size
        ):
            continue

        selected = d.iloc[
            chosen
        ]

        signature = tuple(
            sorted(
                selected[
                    "player"
                ]
            )
        )

        # Prevent duplicate lineups.
        if any(
            lineup["signature"]
            ==
            signature
            for lineup
            in lineups
        ):
            continue

        lineups.append({

            "signature":
                signature,

            "players":
                list(
                    selected[
                        "player"
                    ]
                ),

            "salary":
                int(
                    selected[
                        "salary"
                    ].sum()
                ),

            "projection":
                float(
                    selected[
                        "dk_proxy"
                    ].sum()
                ),

            "win_equity":
                float(
                    selected[
                        "win_pct"
                    ].sum()
                ),

            "top5_equity":
                float(
                    selected[
                        "top5_pct"
                    ].sum()
                ),
        })

    return pd.DataFrame(
        lineups
    )


# ---------------------------------------------------------------------
# WALK-FORWARD BACKTEST
# ---------------------------------------------------------------------

def backtest():

    results = read_csv(
        "results.csv"
    )

    if results.empty:

        raise RuntimeError(
            "results.csv is required."
        )

    required = {
        "event",
        "date",
        "player",
        "finish"
    }

    missing = (
        required -
        set(results.columns)
    )

    if missing:

        raise RuntimeError(
            "results.csv is missing: "
            +
            ", ".join(
                sorted(missing)
            )
        )

    r = results.copy()

    r["date"] = pd.to_datetime(
        r["date"],
        errors="coerce"
    )

    events = (
        r[
            [
                "event",
                "date"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "date"
        )
    )

    records = []

    for _, event_row in events.iterrows():

        event = event_row["event"]
        date = event_row["date"]

        prior = r[
            r["date"] < date
        ]

        actual = r[
            r["event"] == event
        ].copy()

        if (
            len(prior) < 20
            or
            actual.empty
        ):
            continue

        players = actual[
            ["player"]
        ].drop_duplicates()

        historical_stats = (
            prior
            .groupby("player")[
                STAT_COLS
            ]
            .mean()
            .reset_index()
        )

        players = players.merge(
            historical_stats,
            on="player",
            how="left"
        )

        history = read_csv(
            "course_history.csv"
        )

        holes = read_csv(
            "course_holes.csv"
        )

        weather = read_csv(
            "weather.csv"
        )

        features, _, _ = (
            build_features(
                players,
                prior,
                history,
                holes,
                weather,
                str(event)
            )
        )

        features[
            "predicted_finish_rank"
        ] = (
            features[
                "model_strength"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        actual_finish = actual[
            [
                "player",
                "finish"
            ]
        ].copy()

        actual_finish[
            "finish"
        ] = pd.to_numeric(
            actual_finish[
                "finish"
            ],
            errors="coerce"
        )

        merged = (
            features[
                [
                    "player",
                    "predicted_finish_rank"
                ]
            ]
            .merge(
                actual_finish,
                on="player"
            )
            .dropna()
        )

        if len(merged) > 5:

            records.append({

                "event":
                    event,

                "date":
                    date,

                "players":
                    len(merged),

                "spearman_rank_corr":
                    merged[
                        "predicted_finish_rank"
                    ].corr(
                        merged["finish"],
                        method="spearman"
                    ),

                "rank_mae":
                    np.mean(
                        np.abs(
                            merged[
                                "predicted_finish_rank"
                            ]
                            -
                            merged[
                                "finish"
                            ]
                        )
                    )
            })

    output = pd.DataFrame(
        records
    )

    output.to_csv(
        OUT /
        "walk_forward_backtest.csv",
        index=False
    )

    return output


# ---------------------------------------------------------------------
# COMMAND LINE INTERFACE
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PGA Tour Predictor Pro"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True
        )
    )

    # Prediction.
    prediction_parser = (
        subparsers.add_parser(
            "predict"
        )
    )

    prediction_parser.add_argument(
        "--tournament",
        required=True
    )

    prediction_parser.add_argument(
        "--sims",
        type=int,
        default=25000
    )

    prediction_parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # Lineups.
    lineup_parser = (
        subparsers.add_parser(
            "lineups"
        )
    )

    lineup_parser.add_argument(
        "--tournament",
        required=True
    )

    lineup_parser.add_argument(
        "--sims",
        type=int,
        default=25000
    )

    lineup_parser.add_argument(
        "--lineups",
        type=int,
        default=20
    )

    lineup_parser.add_argument(
        "--salary-cap",
        type=int,
        default=50000
    )

    lineup_parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # Backtest.
    subparsers.add_parser(
        "backtest"
    )

    args = parser.parse_args()

    if args.command == "predict":

        config = Config(
            tournament=args.tournament,
            sims=args.sims,
            seed=args.seed
        )

        predictions = predict(
            config
        )

        print(
            "\nTOP MODEL RESULTS\n"
        )

        display_columns = [
            "player",
            "model_rank",
            "win_pct",
            "top5_pct",
            "top10_pct",
            "top20_pct",
            "make_cut_pct",
            "expected_finish",
            "median_finish",
            "expected_4round_score",
            "dk_proxy",
            "salary",
            "points_per_1k",
            "leverage"
        ]

        available = [
            c
            for c in display_columns
            if c in predictions.columns
        ]

        print(
            predictions[
                available
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

    elif args.command == "lineups":

        config = Config(
            tournament=args.tournament,
            sims=args.sims,
            seed=args.seed,
            salary_cap=args.salary_cap,
            roster_size=6
        )

        predictions = predict(
            config
        )

        lineups = optimize_lineups(
            predictions,
            salary_cap=args.salary_cap,
            roster_size=6,
            count=args.lineups,
            seed=args.seed
        )

        safe_name = (
            args.tournament
            .replace(" ", "_")
        )

        path = (
            OUT /
            f"{safe_name}_lineups.csv"
        )

        lineups.to_csv(
            path,
            index=False
        )

        print(
            "\nGENERATED LINEUPS\n"
        )

        print(
            lineups.to_string(
                index=False
            )
        )

        print(
            f"\nSaved to: {path}"
        )

    elif args.command == "backtest":

        results = backtest()

        print(
            "\nWALK-FORWARD BACKTEST\n"
        )

        print(
            results.to_string(
                index=False
            )
        )


def _running_under_streamlit():
    """Return True when this file is being used as a Streamlit entry point.

    Some existing deployments point Streamlit at pga_predictor_pro.py rather
    than app.py.  V10.8 keeps the CLI engine intact, but routes Streamlit runs
    to the unified dashboard so both historical entry-point configurations
    render the same UI.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_under_streamlit():
        import runpy
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    else:
        main()
