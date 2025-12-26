#  Behavior Drift Observatory (BDO)

An **unsupervised system** designed to detect, explain, and visualize **behavioral change over time** with minimal human supervision.

---

## Motivation

In dynamic real-world systems, behavioral data evolves continuously — but **ground-truth labels for “drift” almost never exist**.  
Most existing approaches rely on supervised models that **quietly fail** when faced with unseen behavioral variations.

BDO was born from a simple yet powerful question:

> _How can we reliably detect behavioral change when we don’t know what “change” looks like beforehand?_

---

## Why This Project?

**Behavior Drift Observatory (BDO)** addresses three fundamental challenges in real-world behavioral monitoring:

**1) Absence of reliable labels:**
Behavioral drift rarely comes with ground truth in production. BDO is designed to operate entirely without labels, making it deployable in environments where supervised approaches fail.

**2) Early detection of gradual behavioral change:**
Many systems only react after behavior has already shifted significantly. BDO focuses on detecting subtle, accumulating deviations before they become obvious failures or risks.

**3) Heterogeneous behavior across entities:**
Population-level thresholds often break down when “normal” varies widely across users or systems. BDO uses entity-specific historical baselines, allowing each entity to be evaluated relative to itself.

By focusing on **relative behavioral change instead of prediction**, BDO enables continuous, scalable monitoring of users, services, or systems as their behavior evolves over time — without relying on predefined labels or static models.

---

## What Problem Does It Solve?

BDO helps answer critical questions about evolving behaviors:

- **Is a user behaving differently than before?**  
- **When did that change begin?**  
- **Which aspects of behavior contributed the most?**

All of this is achieved **without the need for labels**, **hand-tuned thresholds**, or **prior knowledge of drift types** — making BDO a flexible and adaptive solution for real-world systems.

---

##  What Makes This Project Stand Out?

- **Fully unsupervised monitoring:** No labels, retraining cycles, or manual parameter tuning required.  
- **User-relative behavioral baselines:** Detects change per individual to avoid population bias and false positives.  
- **Designed for gradual, real-world drift:** Captures slow, accumulating behavioral changes rather than only abrupt anomalies.  
- **Temporal awareness built in:** Identifies when change begins, not just that it happened. 
- **Separation of detection and interpretation:** Drift scoring and explanation are decoupled, making the system extensible and auditable.

---

##  How It Works 

1. **Aggregating behavioral signals:** Raw event streams are summarized using rolling temporal windows.  
2. **Creating adaptive baselines:** Each user builds a dynamic reference model from their own historical behavior.  
3. **Quantifying deviations:** BDO computes a normalized “drift score” that captures behavioral distance over time.  
4. **Detecting persistent drift:** Sustained changes are isolated as true behavioral drift rather than random noise.  
5. **Explaining root causes:** Contributing behavioral features are ranked and contextualized into intuitive summaries for easy interpretation.

---

## Drift Scoring Mechanism

Behavioral drift in BDO is measured using feature-level normalization, converting behavioral deviations into standardized scores relative to each entity’s own history.

Feature-Level Deviation->

For each behavioral feature f:

z_f = |x_f − μ_f| / (σ_f + ε)

Where:
x_f = current feature value

μ_f = user-specific historical mean

σ_f = user-specific historical standard deviation

ε = small constant to prevent division by zero

The overall **Drift Score** is then computed as the average of all feature-wise deviations:

Drift Score = (1 / N) × Σ z_f

Where:

N = number of behavioral features

z_f = normalized deviation for feature f

This formulation ensures that drift detection remains **completely unsupervised** — no labels, thresholds, or predefined drift categories are needed.

---
##  Drift Interpretation Levels 

BDO turns raw drift scores into clear behavioral states.
Instead of showing raw numbers, the **system highlights behavioral stability, emerging change, and drift trends** in an easy-to-understand way.

| Drift Score Range | System Interpretation       |
|-------------------|-----------------------------|
| < 0.08            | Stable behavior             |
| 0.08 – 0.15       | Minor behavioral change     |
| 0.15 – 0.25       | Ongoing behavioral drift    |
| > 0.25            | Strong behavioral shift     |

---

##  Feature Explainability 

Each behavioral feature in BDO is mapped to a clear, intuitive meaning, allowing drift causes to be understood without requiring statistical or domain expertise.

| Raw Feature             | User-Friendly Meaning                           |
|--------------------------|--------------------------------------------------|
| `session_count`          | Frequency of user interactions or sessions      |
| `avg_session_duration`   | Average length of each user session             |
| `active_hours_entropy`   | Consistency of active hours throughout the day  |
| `action_type_entropy`    | Predictability of actions performed             |
| `inter_day_variability`  | Stability of daily activity patterns            |

Feature importance percentages in the UI represent **relative contribution to detected drift** — not absolute statistical values — ensuring interpretability without technical overload.

---

## Technologies Used

BDO is built using a lightweight yet powerful stack of tools for real-time drift detection, computation, and visualization:

- **Python** — core implementation language.  
- **FastAPI** — backend API for drift computation and model orchestration.  
- **Streamlit** — interactive visualization dashboard.  
- **Pandas / NumPy** — data preprocessing and statistical computation.  
- **Scikit-learn** — validation of unsupervised metrics only (no supervised training).  
- **Altair** — clean, declarative plots for behavioral drift visualization.

---

## System Pipeline Overview

| Stage | Description |
|--------|--------------|
| **1. Raw Behavioral Data** | Source logs representing user interactions and events. |
| **2. Rolling Time Aggregation** | Aggregates activity over rolling time windows (e.g., 14 days) to capture temporal patterns. |
| **3. User-Specific Baseline Construction** | Builds adaptive baselines from each user’s historical behavior. |
| **4. Behavior Representation Layer** | Transforms aggregated metrics into numerical feature vectors for drift scoring. |
| **5. Drift Scoring Engine** | Computes normalized drift scores to identify deviation strength. |
| **6. Drift Explanation Engine** | Translates statistical shifts into interpretable feature-level explanations. |
| **7. API Layer (FastAPI)** | Provides REST endpoints for drift computation and data retrieval. |
| **8. Visualization & Insights Dashboard (Streamlit)** | Presents human-readable drift insights and behavior trends through interactive dashboards. |


---
##  Installation & Running the Project

1. **Clone the repository:**:
git clone https://github.com/AryaSingh16/Behavior-Drift-Observatory.git

2. **Install dependencies**:
pip install -r requirements.txt

3. **Run the FastAPI backend:**
uvicorn src.api.main:app --reload

4. **Launch the Streamlit dashboard:**
streamlit run src/frontend/app.py

The FastAPI server provides drift scoring APIs, while the Streamlit interface visualizes user-level behavioral changes in real time.

## Access

Once running locally, the system components can be accessed via:

- **API Endpoint:** [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- **Web UI (Dashboard):** [http://localhost:8501](http://localhost:8501)

---

## Credits

**Behavior Drift Observatory (BDO)** was created and developed by **Arya A. Singh**.  

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.
