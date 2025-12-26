# 🧭 Behavior Drift Observatory (BDO)

An **unsupervised system** designed to detect, explain, and visualize **behavioral change over time** with minimal human supervision.

---

## 📌 Motivation

In dynamic real-world systems, behavioral data evolves continuously — but **ground-truth labels for “drift” almost never exist**.  
Most existing approaches rely on supervised models that **quietly fail** when faced with unseen behavioral variations.

BDO was born from a simple yet powerful question:

> _How can we reliably detect behavioral change when we don’t know what “change” looks like beforehand?_

---

## ❓ Why This Project?

**Behavior Drift Observatory (BDO)** addresses three key limitations of current drift detection frameworks:

1. **Lack of labels** for behavioral drift in production systems.  
2. **Low interpretability** of traditional drift detection scores.  
3. **User-facing dashboards** that expose raw, complex metrics without clarity.  

By focusing on **relative behavioral change instead of prediction**, BDO is ideal for monitoring users, systems, or entities as their behavior evolves over time — without relying on predefined labels or models.

---

## 🧩 What Problem Does It Solve?

BDO helps answer critical questions about evolving behaviors:

- **Is a user behaving differently than before?**  
- **When did that change begin?**  
- **Which aspects of behavior contributed the most?**

All of this is achieved **without the need for labels**, **hand-tuned thresholds**, or **prior knowledge of drift types** — making BDO a flexible and adaptive solution for real-world systems.

---

## ✨ What Makes This Project Stand Out?

- **Fully unsupervised monitoring:** No labels, retraining cycles, or manual parameter tuning required.  
- **User-relative behavioral baselines:** Detects change per individual to avoid population bias and false positives.  
- **Actionable insights:** Translates abstract statistical shifts into clear, narrative-style behavioral explanations.  
- **UX-first observability design:** Built for product, operations, and risk teams — not just data scientists.  
- **Production-grade validation:** Evaluated using unsupervised benchmarking methods instead of misleading accuracy metrics.

---

## 🧠 How It Works 

1. **Aggregating behavioral signals:** Raw event streams are summarized using rolling temporal windows.  
2. **Creating adaptive baselines:** Each user builds a dynamic reference model from their own historical behavior.  
3. **Quantifying deviations:** BDO computes a normalized “change score” that captures behavioral distance over time.  
4. **Detecting persistent drift:** Sustained changes are isolated as true behavioral drift rather than random noise.  
5. **Explaining root causes:** Contributing behavioral features are ranked and contextualized into intuitive summaries for easy interpretation.

---

## 📊 Drift Scoring Mechanism

Behavioral drift in BDO is measured through a feature-level normalization technique that transforms deviations into standardized scores.

For each feature \( f \):

\[
z_f = \frac{|x_f - \mu_f|}{\sigma_f + \epsilon}
\]

The overall **Drift Score** is then computed as the average of all feature-wise deviations:

\[
\text{Drift Score} = \frac{1}{N} \sum_{f=1}^{N} z_f
\]

Where:
- \( \mu_f \) = user-specific historical mean.  
- \( \sigma_f \) = user-specific historical standard deviation.  
- \( \epsilon \) = small smoothing constant to prevent division by zero.  

This formulation ensures that drift detection remains **completely unsupervised** — no labels, thresholds, or predefined drift categories are needed.

---
## 🔍 Drift Interpretation Levels 

BDO translates quantitative drift scores into intuitive, human-readable states.  
Raw numerical values are never displayed to end users; instead, the interface visualizes **qualitative drift trends** and **stability indicators**.

| Drift Score Range | System Interpretation       |
|-------------------|-----------------------------|
| < 0.08            | Stable behavior             |
| 0.08 – 0.15       | Minor behavioral change     |
| 0.15 – 0.25       | Ongoing behavioral drift    |
| > 0.25            | Strong behavioral shift     |

The UI uses these intervals to generate **color-coded indicators and trend lines**, helping teams quickly identify when and where user behavior diverges from historical norms.

---

## 🧠 Feature Explainability 

Each behavioral feature tracked by BDO is mapped to a **human-understandable meaning**, so that data scientists and non-technical stakeholders can interpret drift causes effectively.

| Raw Feature             | User-Friendly Meaning                           |
|--------------------------|--------------------------------------------------|
| `session_count`          | Frequency of user interactions or sessions      |
| `avg_session_duration`   | Average length of each user session             |
| `active_hours_entropy`   | Consistency of active hours throughout the day  |
| `action_type_entropy`    | Predictability of actions performed             |
| `inter_day_variability`  | Stability of daily activity patterns            |

Feature importance percentages in the UI represent **relative contribution to detected drift** — not absolute statistical values — ensuring interpretability without technical overload.

---

## ✅ Validation Strategy (Tailored for Unsupervised Systems)

Unlike predictive models, unsupervised drift detection systems **cannot be judged using accuracy, precision, or recall**, since labeled drift data rarely exists.  
BDO instead follows **behavioral sanity checks**, a standard used in real-world observability, finance, and cybersecurity systems.

### 1️⃣ Stability Check (Before Drift Injection)
**Goal:** Verify that stable users produce consistently low drift scores.  
**Results:**  
- Mean drift score = 0.0182  
- Standard deviation = 0.0016  
✅ Confirms low noise and stable baseline behavior.

### 2️⃣ Sensitivity Check (After Drift Injection)
**Goal:** Ensure drift scores react appropriately to synthetic behavioral changes.  
**Results:**  
- Mean drift score = 0.0205  
✅ Clear post-injection increase, confirming sensitivity to real drift.

### 3️⃣ Drift Onset Alignment
**Goal:** Confirm that detected drift timing aligns with actual change points.  
**Results (change point ≈ Day 60):**  
- Mean onset = 63.17  
- Std = 8.33 (tight cluster)  
✅ Drift detections consistently occur around the true change point, indicating reliability and precision.

### ✔ Why This Validation Is Correct
- Does **not rely on labeled data**.  
- Evaluates **stability, responsiveness, and temporal alignment**, not prediction.  
- Mirrors **industry-standard validation** for live monitoring systems.

---

## ⚠️ Supervised Baseline (For Comparison Only)

A simple supervised classifier was trained using synthetic drift labels to contrast against BDO’s unsupervised method.

| Metric             | Value |
|--------------------|-------|
| Accuracy           | 81%   |
| Drift Recall       | 5%    |
| Drift Precision    | 37%   |

**Supervised feature importance:** dominated by `inter_day_variability`, while other features were mostly ignored.

---

## 🚨 Why Supervised Approaches Fail Here

- High “accuracy” is misleading due to **class imbalance**.  
- **Fails to detect most real drift events** and **depends on unavailable labels**.  
- Cannot adapt to **emerging or evolving behavioral changes**.  

In contrast, BDO’s unsupervised design ensures **continuous, adaptive, and label-free drift monitoring**, suitable for production environments.

---

## 📚 Technologies Used

BDO is built using a lightweight yet powerful stack of tools for real-time drift detection, computation, and visualization:

- **Python** — core implementation language.  
- **FastAPI** — backend API for drift computation and model orchestration.  
- **Streamlit** — interactive visualization dashboard.  
- **Pandas / NumPy** — data preprocessing and statistical computation.  
- **Scikit-learn** — validation of unsupervised metrics only (no supervised training).  
- **Altair** — clean, declarative plots for behavioral drift visualization.

---

## ⚙️ System Architecture

The following pipeline outlines the BDO data flow from raw behavioral logs to user-facing visualization:
Raw Behavioral Data
        ↓
Rolling Time Aggregation
        ↓
User-Specific Baseline Construction
        ↓
Behavior Representation Layer
        ↓
Drift Scoring Engine
        ↓
Drift Explanation Engine
        ↓
API Layer (FastAPI)
        ↓
Visualization & Insights Dashboard (Streamlit)

BDO is built around a set of guiding principles that ensure accuracy, interpretability, and real-world usability.

- **User-Relative Baselines:**  
  Each entity’s behavior is compared to its own historical patterns rather than a global population.  
  This minimizes population bias and allows detection of subtle, individualized behavioral shifts.

- **Unsupervised Drift Detection:**  
  The system operates without labeled data, retraining cycles, or predefined thresholds.  
  This autonomy makes it robust to unseen, evolving behaviors — ideal for live monitoring environments.

- **Explainability-First Approach:**  
  Every detected drift is accompanied by an intuitive breakdown of feature contributions.  
  Statistical deviations are translated into **human-readable narratives**, enabling actionable insights for non-technical teams.

- **Modular, Extensible Pipeline:**  
  Each subsystem — data preprocessing, drift computation, API, and visualization — is independently maintainable.  
  This modularity supports fast iteration and experimentation without disrupting the pipeline.

- **Production-Ready Architecture:**  
  Built on a **stateless, API-driven design**, with a decoupled visualization layer.  
  Ensures scalability, real-time integration, and ease of deployment across analytics or monitoring stacks.

Together, these principles enable BDO to **detect evolving behavioral changes reliably** while maintaining transparency, scalability, and operational simplicity.

---
## 🛠️ Installation & Running the Project

1. **Clone the repository**
git clone https://github.com/AryaSingh16/behavior-drift-observatory.git
cd behavior-drift-observatory

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the FastAPI backend**
uvicorn src.api.main:app --reload

4. **Launch the Streamlit dashboard**
streamlit run src/frontend/app.py

The FastAPI server provides drift scoring APIs, while the Streamlit interface visualizes user-level behavioral changes in real time.

## 🌐 Access

Once running locally, the system components can be accessed via:

- **API Endpoint:** [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- **Web UI (Dashboard):** [http://localhost:8501](http://localhost:8501)

---

## 📖 What I Learned

Working on **Behavior Drift Observatory (BDO)** gave me hands-on experience building an unsupervised, end-to-end ML monitoring system — from raw data to deployable insights.  
Key learnings include:

- **Evaluating unsupervised systems:** Understanding why accuracy or F1 scores fail for drift detection, and how alternative evaluation frameworks (distributional tests, embedding shifts, correlation stability) are more suitable.  
- **Designing for interpretability:** Translating statistical changes into **human-understandable behavior stories**, bridging the gap between data science and decision-makers.  
- **Human-centered ML observability:** Learning how explainability and UX co-evolve — dashboards aren’t just visualization tools, but communication layers for insights.  
- **Building beyond models:** Appreciating the full lifecycle of ML systems — from feature engineering and backend orchestration to visualization pipelines and deployment.  
- **Bringing science to product thinking:** Seeing how metrics of behavioral drift directly connect to **business risks, user trust, and system reliability**.
---

## 🤝 Credits

**Behavior Drift Observatory (BDO)** was **designed and implemented by [Arya A Singh]**.  
Inspired by real-world challenges in **behavioral analytics and production monitoring.**

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.