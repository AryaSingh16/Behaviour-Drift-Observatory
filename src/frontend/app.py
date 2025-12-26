import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt

API_BASE = "http://127.0.0.1:8000"
DRIFT_THRESHOLD = 0.15
CONSECUTIVE_DAYS = 3

st.set_page_config(
    page_title="Behavior Drift Observatory",
    layout="wide",
)
def drift_level(score):
    if score < 0.08:
        return 0  # Stable
    elif score < 0.15:
        return 1  # Slight change
    elif score < 0.25:
        return 2  # Noticeable change
    else:
        return 3  # Strong change

# --------------------------------------------
# User-facing behavior descriptions
# --------------------------------------------
FEATURE_UI_LABELS = {
    "session_count": "Activity frequency",
    "avg_session_duration": "Session length",
    "active_hours_entropy": "Consistency of active hours",
    "action_type_entropy": "Predictability of actions",
    "inter_day_variability": "Day-to-day activity stability",
}

FEATURE_EXPLANATIONS = {
    "session_count": "The user became active more or less often",
    "avg_session_duration": "Session lengths changed compared to the past",
    "active_hours_entropy": "The timing of activity became more or less consistent",
    "action_type_entropy": "User actions became more or less predictable",
    "inter_day_variability": "Daily activity levels fluctuated more than usual",
}

FEATURE_METADATA = {
    "inter_day_variability": {
        "label": "Daily activity consistency",
        "description": "How similar the user’s activity level is from one day to the next",
    },
    "user_entropy": {
        "label": "Usage predictability",
        "description": "How predictable the user’s overall usage patterns are",
    },
    "session_frequency": {
        "label": "Activity frequency",
        "description": "How often the user is active over time",
    },
    "time_of_day_shift": {
        "label": "Typical active hours",
        "description": "When during the day the user is usually active",
    },
}


def friendly_feature_phrase(feature, direction):
    meta = FEATURE_METADATA.get(feature)

    if not meta:
        return "a general usage pattern change"

    base = meta["label"].lower()

    if direction == "increase":
        return f"an increase in {base}"
    else:
        return f"a decrease in {base}"



def drift_to_percentage(score):
    """
    Convert internal drift score to user-facing percentage.
    Calibrated to dataset distribution.
    """
    pct = min(score / 0.3, 1.0) * 100
    return int(round(pct))


def is_major_drift(score):
    """
    Define what counts as a major drift event.
    """
    return score >= 0.2

def behavior_label(score):
    if score < 0.08:
        return "Stable"
    elif score < 0.15:
        return "Minor Drift"
    elif score < 0.25:
        return "Drifting"
    else:
        return "Strong Drift"


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Behavior Drift Observatory")
st.caption("Unsupervised system for detecting and explaining behavioral change")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Entity Selection")

user_id = st.sidebar.text_input(
    "User ID",
    value="user_320",
)

# -------------------------------------------------
# Fetch timeline
# -------------------------------------------------
try:
    timeline_resp = requests.get(f"{API_BASE}/drift/score/{user_id}").json()
    timeline_df = pd.DataFrame(timeline_resp["timeline"])
except Exception:
    st.error("User not found or API unavailable")
    st.stop()

# -------------------------------------------------
# Compute drift onset
# -------------------------------------------------
timeline_df = timeline_df.sort_values("day")

timeline_df["Behavior State"] = timeline_df["drift_score"].apply(
    behavior_label
)

timeline_df["Day Label"] = timeline_df["day"].apply(
    lambda d: f"Day {d}"
)


above = timeline_df["drift_score"] > DRIFT_THRESHOLD
onset_day = None

for i in range(len(above) - CONSECUTIVE_DAYS):
    if above.iloc[i:i+CONSECUTIVE_DAYS].all():
        onset_day = int(timeline_df.iloc[i]["day"])
        break

latest_day = int(timeline_df.iloc[-1]["day"])
latest_score = float(timeline_df.iloc[-1]["drift_score"])

# -------------------------------------------------
# Drift status
# -------------------------------------------------
st.subheader("Drift Status")

col1, col2, col3 = st.columns(3)


if (latest_score < 0.08):
    status = "Stable"
elif (latest_score >= 0.08 and latest_score < 0.15):
    status = "Minor Drift"
else:
    status = "Drifting" 



col1.metric("Entity", user_id)
col2.metric("Current Status", status)
current_pct = drift_to_percentage(latest_score)

col3.metric(
    "Current Behavior Change",
    f"{current_pct}%",
)


# -------------------------------------------------
# Onset info
# -------------------------------------------------
st.subheader("Drift Onset")

if onset_day is not None:
    st.success(f"Behavioral drift detected starting around **day {onset_day}**")
else:
    st.info("No sustained behavioral drift detected")

# -------------------------------------------------
# Drift timeline
# -------------------------------------------------

st.subheader("Behavior Change Over Time")

st.caption(
    "Hover over the line to see daily behavior interpretation"
)

chart = (
    alt.Chart(timeline_df)
    .mark_line(point=True,interpolate="monotone")
    .encode(
        x=alt.X(
            "day:Q",
            title="Day",
            axis=alt.Axis(grid=False)
        ),
        y=alt.Y(
            "drift_score:Q",
            title=None,                 # 🚫 NO Y-axis label
            axis=alt.Axis(labels=False, ticks=True, grid=True)
        ),
        tooltip=[
            alt.Tooltip("Day Label:N", title="Time"),
            alt.Tooltip("Behavior State:N", title="Behavior")
        ]
    )
    .properties(height=300)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)    


# -------------------------------------------------
# Explanation
# -------------------------------------------------

st.subheader("Behavior Insights")

explain_resp = requests.get(f"{API_BASE}/drift/explanation/{user_id}").json()
explain_df = pd.DataFrame(explain_resp["explanations"])
raw_explain_df = explain_df.copy()

raw_explain_df["abs_contribution"] = raw_explain_df["contribution"].abs()

top_raw_row = (
    raw_explain_df
    .sort_values("abs_contribution", ascending=False)
    .iloc[0]
)

top_raw_feature = top_raw_row["feature"]
top_raw_direction = top_raw_row["direction"]

total_contribution = raw_explain_df["abs_contribution"].sum()

top_raw_impact = int(
    round(
        top_raw_row["abs_contribution"] / total_contribution * 100
    )
)

# -----------------------------
# User-friendly behavior labels
# -----------------------------
explain_df["Behavior Aspect"] = explain_df["feature"].map(
    FEATURE_UI_LABELS
).fillna("Overall usage behavior")

# Direction wording
explain_df["Change"] = explain_df["direction"].map({
    "increase": "Increased",
    "decrease": "Decreased"
})

# -----------------------------
# Normalize contribution → %
# -----------------------------
total_contribution = explain_df["contribution"].abs().sum()

if total_contribution > 0:
    explain_df["Impact (%)"] = (
        explain_df["contribution"].abs() / total_contribution * 100
    ).round().astype(int)
else:
    explain_df["Impact (%)"] = 0

# -----------------------------

explain_df["abs_contribution"] = explain_df["contribution"].abs()

explain_df = (
    explain_df
    .sort_values("abs_contribution", ascending=False)
    .groupby("Behavior Aspect", as_index=False)
    .first()
)

explain_df["Observed Pattern"] = (
    explain_df["Change"] + " (" +
    explain_df["Impact (%)"].astype(str) + "%)"
)

explain_df = explain_df[[
    "Behavior Aspect",
    "Observed Pattern"
]]


st.dataframe(explain_df, use_container_width=True, hide_index=True)


# -------------------------------------------------
# Human summary
# -------------------------------------------------
st.subheader("System Interpretation")

top = explain_df.iloc[0]
current_level = drift_level(latest_score)

LEVEL_TEXT = {
    0: "stable behavior",
    1: "minor behavioral changes",
    2: "noticeable behavioral shift",
    3: "strong behavioral change"
}

severity_text = LEVEL_TEXT[current_level]

top_feature_phrase = friendly_feature_phrase(
    top_raw_feature,
    top_raw_direction
)

if current_level == 0:
    # ✅ STABLE USER INTERPRETATION
    st.markdown(
        f"""
**Summary**
- The user currently shows **{severity_text}**
- Activity patterns remain consistent over time with only natural fluctuations
- No sustained behavioral shift has been detected when compared to historical behavior
"""
    )
elif current_level == 1:  # Minor Drift
    st.markdown(
        f"""
**Summary**
- The user shows **early signs of behavioral change**
- No sustained shift has been detected yet
- The affected aspect is **{top_feature_phrase}** (~{top_raw_impact}% influence)
- The system continues monitoring for sustained change
"""
    )

else:
    # ✅ DRIFTING USER INTERPRETATION
    st.markdown(
        f"""
**Summary**
- The user currently shows **{severity_text}**
- Behavioral change started around day {onset_day if onset_day else "N/A"}
- The primary driver of this change was **{top_feature_phrase}** (~{top_raw_impact}% influence)
- Assessment is based on comparison with the user’s own historical behavior
"""
    )