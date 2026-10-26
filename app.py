"""
Campus Facility Reservation and Monitoring System - Data Mining Dashboard
Streamlit app for the SAD Data Mining Laboratory Examination.

Author : April Jhon B. Relampago (BSIS 3A)
Run    : streamlit run app.py
"""
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
st.set_page_config(page_title="Facility Reservation Analytics", page_icon="🏫", layout="wide")

BASE = Path(__file__).parent
RAW_CSV = BASE / "facility_reservations.csv"
CLEAN_CSV = BASE / "facility_reservations_clean.csv"
MODEL_PATH = BASE / "trained_model.joblib"
DIAGRAM_PNG = BASE / "Use_Case_Diagram.png"

RANDOM_STATE = 42
CAT_FEATURES = ["Facility_Type", "User_Category", "Time_Slot", "Day_Type"]
NUM_FEATURES = ["Participants", "Lead_Time_Days", "Duration_Hours"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "Reservation_Outcome"
POSITIVE = "Not Completed"
LABELS = ["Completed", "Not Completed"]
LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]

FACILITIES = ["Classroom", "Computer Lab", "Auditorium", "Sports Court", "Conference Room", "Study Room"]
USERS = ["Student", "Student Organization", "Faculty", "Staff", "External Guest"]
SLOTS = ["Morning", "Afternoon", "Evening"]

OBS, PRED, RED, GREEN = "#4C78A8", "#F28E2B", "#E45756", "#59A14F"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- data
@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)


def build_pipeline() -> Pipeline:
    """Same Decision Tree pipeline used in the laboratory notebook."""
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=8,
                                        class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def original_feature(name: str) -> str:
    for c in CAT_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


@st.cache_resource
def get_bundle() -> dict:
    """Load trained_model.joblib (or retrain if it cannot be loaded) and rebuild the same 80/20 split."""
    clean = load_clean()
    model_df = clean[clean[TARGET].notna()].reset_index(drop=True)
    X, y = model_df[FEATURES], model_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    source = "trained_model.joblib"
    try:
        pipe = joblib.load(MODEL_PATH)
        pipe.predict(X_test.head(2))                      # sanity check (also catches version problems)
    except Exception:
        pipe = build_pipeline().fit(X_train, y_train)
        source = "re-trained at start-up (saved model could not be loaded)"

    tree = pipe.named_steps["tree"]
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = (pd.Series(tree.feature_importances_, index=[original_feature(n) for n in names])
           .groupby(level=0).sum().sort_values(ascending=False))
    return dict(pipe=pipe, tree=tree, names=names, importance=imp, source=source, model_df=model_df,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def risk_level(p: float) -> str:
    return "Low" if p <= 0.35 else ("Medium" if p <= 0.60 else "High")


def risk_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, list(pipe.classes_).index(POSITIVE)]


def explain_prediction(bundle: dict, row_df: pd.DataFrame) -> list:
    """Human-readable decision-path rules behind one prediction."""
    pipe, tree, names = bundle["pipe"], bundle["tree"], bundle["names"]
    Xt = pipe.named_steps["prep"].transform(row_df)
    rules = []
    for nid in tree.decision_path(Xt).indices:
        if tree.tree_.children_left[nid] == tree.tree_.children_right[nid]:
            continue                                       # leaf node
        f, thr = tree.tree_.feature[nid], tree.tree_.threshold[nid]
        val, nm = Xt[0, f], names[f]
        if nm in NUM_FEATURES:
            rules.append(f"{nm} {'<=' if val <= thr else '>'} {thr:.1f}")
        else:
            orig = original_feature(nm)
            rules.append(f"{orig} {'is' if val > thr else 'is not'} {nm[len(orig) + 1:]}")
    return rules


def safe_metrics(y_true, y_hat) -> dict:
    n_true, n_pred = int((y_true == POSITIVE).sum()), int((y_hat == POSITIVE).sum())
    return dict(
        acc=accuracy_score(y_true, y_hat),
        prec=precision_score(y_true, y_hat, pos_label=POSITIVE, zero_division=0) if n_pred else np.nan,
        rec=recall_score(y_true, y_hat, pos_label=POSITIVE, zero_division=0) if n_true else np.nan,
        f1=f1_score(y_true, y_hat, pos_label=POSITIVE, zero_division=0) if (n_true and n_pred) else np.nan,
        n_true=n_true, n_pred=n_pred,
    )


def fmt(x) -> str:
    return "undefined" if pd.isna(x) else f"{x:.2f}"


# --------------------------------------------------------------------------- EDA tables
def eda_tables(df: pd.DataFrame) -> dict:
    known = df[df["Facility_Type"] != "Unknown"]
    fac_counts = known["Facility_Type"].value_counts()

    eda = df[df[TARGET].notna()].copy()
    user_tbl = (eda[eda["User_Category"] != "Unknown"].groupby("User_Category")[TARGET]
                .agg(n="size", not_completed=lambda s: int((s == POSITIVE).sum())))
    user_tbl["pct_not_completed"] = (100 * user_tbl["not_completed"] / user_tbl["n"]).round(1)
    user_tbl = user_tbl.sort_values("pct_not_completed", ascending=False)

    lead_eda = eda[eda["Lead_Time_Category"].isin(LEAD_ORDER)]
    lead_ct = pd.crosstab(lead_eda["Lead_Time_Category"], lead_eda[TARGET]).reindex(LEAD_ORDER)
    lead_ct["n"] = lead_ct.sum(axis=1)
    lead_ct["pct_not_completed"] = (100 * lead_ct[POSITIVE] / lead_ct["n"]).round(1)

    return dict(fac_counts=fac_counts, n_fac=int(fac_counts.sum()), eda=eda, n_out=len(eda),
                user_tbl=user_tbl, lead_ct=lead_ct,
                hi_l=lead_ct["pct_not_completed"].idxmax(), lo_l=lead_ct["pct_not_completed"].idxmin())


def chart_facility(t, ax):
    bars = ax.bar(t["fac_counts"].index, t["fac_counts"].values, color=OBS)
    ax.set_title(f"Reservations by Facility Type\n(n = {t['n_fac']} with known facility)", fontsize=10)
    ax.set_ylabel("Number of reservations"); ax.tick_params(axis="x", rotation=30)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, int(b.get_height()), ha="center", fontsize=8)


def chart_user(t, ax):
    u = t["user_tbl"]
    bars = ax.bar(u.index, u["pct_not_completed"], color=RED)
    ax.set_title("Cancellation / No-Show Rate by User Category\n(denominator = known outcome per category)", fontsize=10)
    ax.set_ylabel("% Not Completed"); ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(60, u["pct_not_completed"].max() + 12))
    for b, (n, k, p) in zip(bars, u[["n", "not_completed", "pct_not_completed"]].values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{p:.1f}%\n({int(k)}/{int(n)})", ha="center", fontsize=8)


def chart_lead(t, ax):
    ct = t["lead_ct"]
    pc, pn = 100 * ct["Completed"] / ct["n"], 100 * ct[POSITIVE] / ct["n"]
    ax.bar(LEAD_ORDER, pc, color=GREEN, label="Completed")
    ax.bar(LEAD_ORDER, pn, bottom=pc, color=RED, label="Not Completed")
    ax.set_title("Outcome by Lead-Time Category\n(100% stacked)", fontsize=10)
    ax.set_ylabel("% of reservations in category")
    ax.set_xticks(range(len(LEAD_ORDER)))
    ax.set_xticklabels([f"{o}\n(n={int(ct.loc[o, 'n'])})" for o in LEAD_ORDER], fontsize=8)
    for i, o in enumerate(LEAD_ORDER):
        ax.text(i, pc[o] + pn[o] / 2, f"{pn[o]:.1f}%", ha="center", color="white", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)


# --------------------------------------------------------------------------- pages
def page_overview():
    st.title("🏫 Campus Facility Reservation and Monitoring System")
    st.caption("Laboratory Examination - Data Mining and Systems Analysis Integration  |  "
               "Systems Analysis and Design (SAD)  |  BS Information Systems")
    df, raw, b = load_clean(), load_raw(), get_bundle()
    t = eda_tables(df)
    m = safe_metrics(b["y_test"], b["pipe"].predict(b["X_test"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", len(raw))
    c2.metric("Clean, unique records", len(df))
    c3.metric("Not-completed rate", f"{100 * (t['eda'][TARGET] == POSITIVE).mean():.1f}%",
              help=f"Cancelled + No-Show, out of {t['n_out']} reservations with a known outcome")
    c4.metric("Model accuracy (test)", f"{m['acc']:.2f}")

    st.markdown("""
**Pipeline:** historical data → cleaning → analysis → prediction → evidence-based requirements → use cases → decision support.

Use the **sidebar** to move through the parts of the examination:

| Page | Content |
|---|---|
| Data & Cleaning | Dataset, data dictionary, data-quality problems and how they were fixed (Tasks 1-5) |
| Exploratory Analysis | Business questions with charts (Task 6) |
| Model & Reliability | Decision Tree, metrics, confusion matrix, reliability verdict (Tasks 7-8) |
| Risk Predictor | Try the model on a new reservation (decision support) |
| Admin Dashboard | Observed facts vs. predicted risk (Task 11) |
| Requirements & Use Case | Evidence-based requirements and updated UML diagram (Tasks 9-10) |
| Critical Analysis | Short written answers (Task 12) |
""")
    st.info("All data is **synthetic** - no real student information is used. "
            "Predictions are probabilities that support, never replace, the administrator's decision.")


def page_data():
    st.header("Data & Cleaning (Tasks 1-5)")
    raw, df = load_raw(), load_clean()

    tab1, tab2, tab3, tab4 = st.tabs(["Raw data", "Data dictionary", "Cleaning decisions", "Clean data"])

    with tab1:
        st.write(f"**{raw.shape[0]} rows x {raw.shape[1]} columns**")
        st.dataframe(raw.head(25))
        st.subheader("Summary statistics (numerical)")
        st.dataframe(raw.describe())
        st.caption("The raw summary already exposes problems: negative or 999 `Participants`, negative "
                   "`Lead_Time_Days`, `Duration_Hours` up to 24, negative attendance, and uneven counts (missing values).")
        st.download_button("⬇ Download raw CSV", raw.to_csv(index=False).encode(), "facility_reservations.csv", "text/csv")

    with tab2:
        st.dataframe(pd.DataFrame([
            ["Reservation_ID", "Identifier", "Unique reservation code", "Identifier (not used)"],
            ["Facility_Type", "Categorical", "Type of facility reserved", "Predictor"],
            ["User_Category", "Categorical", "Who made the request", "Predictor"],
            ["Request_Date", "Date", "Date the request was submitted", "Used to derive lead time"],
            ["Reservation_Date", "Date", "Date the facility is reserved for", "Used to derive weekday/weekend"],
            ["Lead_Time_Days", "Numerical", "Days between request and reserved date", "Predictor"],
            ["Time_Slot", "Categorical", "Morning / Afternoon / Evening", "Predictor"],
            ["Duration_Hours", "Numerical", "Requested duration of use", "Predictor"],
            ["Participants", "Numerical", "Declared number of participants", "Predictor"],
            ["Usage_Status", "Categorical", "Used / Cancelled / No-Show (known only AFTER the event)", "Source of target - NOT a predictor"],
            ["Actual_Attendance", "Numerical", "Persons who actually attended", "Post-outcome - NOT a predictor"],
            ["Cancellation_Days_Before", "Numerical", "Days before the event a cancellation was made", "Post-outcome - NOT a predictor"],
        ], columns=["Variable", "Type", "Description", "Role"]), hide_index=True)
        st.markdown("**Target:** `Reservation_Outcome` = *Completed* (Used) vs *Not Completed* (Cancelled or No-Show).  \n"
                    "**Leakage guard:** `Usage_Status`, `Actual_Attendance`, `Attendance_Ratio` and "
                    "`Cancellation_Days_Before` are only known after the outcome, so they are never used as predictors.")

    with tab3:
        st.markdown("""
*Principle: never invent values the record does not support. The raw data is preserved; cleaning is done on a copy.*

| Problem | Decision |
|---|---|
| Inconsistent categories (`Class Room`, `CLASSROOM`, `Canceled`...) | Trim, ignore case/punctuation, map to one official label |
| Duplicate transactions | Removed **after** standardising (near-duplicates differ only by case) |
| Missing `Facility_Type` / `User_Category` | Labelled `Unknown` (not guessed) |
| Missing `Usage_Status` | `Reservation_Outcome` stays missing; excluded from modelling |
| Invalid `Participants` (<= 0 or > 300) | Set to missing |
| Invalid `Duration_Hours` (< 1 or > 8) | Set to missing |
| Negative `Actual_Attendance` | Set to missing |
| Lead time | Recomputed from the two dates when valid; missing if dates conflict |
| Missing numerics | Imputed with the **training-set median inside the model pipeline** only |
""")
        dq = pd.DataFrame({"Raw missing": raw.isna().sum(),
                           "Clean missing (NaN)": df.reindex(columns=raw.columns).isna().sum()})
        dq["Clean labelled 'Unknown'"] = [int((df[c] == "Unknown").sum()) if c in ("Facility_Type", "User_Category") else 0
                                          for c in dq.index]
        st.subheader("Missing values per column: raw vs clean")
        st.dataframe(dq)
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", len(raw))
        c2.metric("Exact duplicates in raw", int(raw.duplicated().sum()))
        c3.metric("Clean rows", len(df))

    with tab4:
        st.write(f"**{df.shape[0]} rows x {df.shape[1]} columns** - includes derived variables "
                 "`Lead_Time_Category`, `Reservation_Outcome`, `Attendance_Ratio`, `Reservation_Weekday`, `Day_Type`.")
        st.dataframe(df)
        st.download_button("⬇ Download clean CSV", df.to_csv(index=False).encode(),
                           "facility_reservations_clean.csv", "text/csv")


def page_eda():
    st.header("Exploratory Data Analysis (Task 6)")
    df = load_clean()
    t = eda_tables(df)
    st.caption("Percentages always state their denominator. Rows with an `Unknown` category or a missing outcome "
               "are excluded from the question they cannot be assigned to.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    chart_facility(t, axes[0]); chart_user(t, axes[1]); chart_lead(t, axes[2])
    plt.tight_layout()
    st.pyplot(fig)

    fc, ut, lc = t["fac_counts"], t["user_tbl"], t["lead_ct"]
    top_fac, hi_u, lo_u = fc.index[0], ut.iloc[0], ut.iloc[-1]
    hi_l, lo_l = t["hi_l"], t["lo_l"]
    st.subheader("Findings")
    st.markdown(f"""
- **Q1 - Facility with the most reservations:** **{top_fac}** with {int(fc.iloc[0])} of {t['n_fac']} reservations
  ({100 * fc.iloc[0] / t['n_fac']:.1f}%). {len(df) - t['n_fac']} reservations with 'Unknown' facility were excluded.
- **Q2 - Highest cancellation/no-show rate:** **{ut.index[0]}** - {int(hi_u['not_completed'])} of {int(hi_u['n'])}
  = {hi_u['pct_not_completed']:.1f}%, versus {lo_u['pct_not_completed']:.1f}% for {ut.index[-1]}.
- **Q3 - Lead time vs outcome:** the not-completed rate is highest for **{hi_l}**
  ({lc.loc[hi_l, 'pct_not_completed']:.1f}% of {int(lc.loc[hi_l, 'n'])}) and lowest for **{lo_l}**
  ({lc.loc[lo_l, 'pct_not_completed']:.1f}% of {int(lc.loc[lo_l, 'n'])}).
""")
    if (ut["n"] < 30).any():
        st.warning("Small groups (n < 30): " + ", ".join(ut.index[ut["n"] < 30]) + " - percentages are less stable.")
    st.caption("These are associations in this dataset, not proof of causation.")

    with st.expander("Show the tables behind the charts"):
        st.write("Reservations per facility type"); st.dataframe(fc.rename("reservations"))
        st.write("Not Completed by user category"); st.dataframe(ut)
        st.write("Outcome by lead-time category"); st.dataframe(lc)


def page_model():
    st.header("Decision Tree Model & Reliability (Tasks 7-8)")
    b = get_bundle()
    pipe, X_test, y_test = b["pipe"], b["X_test"], b["y_test"]
    y_pred = pipe.predict(X_test)
    m = safe_metrics(y_test, y_pred)
    majority = y_test.value_counts(normalize=True).max()
    train_acc = accuracy_score(b["y_train"], pipe.predict(b["X_train"]))

    st.caption(f"Model source: {b['source']}  |  {len(b['model_df'])} rows used "
               f"(train {len(b['X_train'])} / test {len(X_test)}, stratified 80/20, random_state = 42)")
    st.markdown("**7 predictors known at submission:** " + ", ".join(f"`{f}`" for f in FEATURES))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['acc']:.3f}", f"baseline {majority:.3f}", delta_color="off")
    c2.metric("Precision (Not Completed)", fmt(m["prec"]))
    c3.metric("Recall (Not Completed)", fmt(m["rec"]))
    c4.metric("F1 (Not Completed)", fmt(m["f1"]))
    st.caption(f"Train accuracy {train_acc:.3f} vs test accuracy {m['acc']:.3f}.")

    left, right = st.columns([1, 1.3])
    with left:
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix (test set)")
        plt.tight_layout(); st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    with right:
        imp = b["importance"]
        fig, ax = plt.subplots(figsize=(6, 4))
        imp[::-1].plot.barh(ax=ax, color=OBS)
        ax.set_title("Feature importance (grouped by original variable)"); ax.set_xlabel("Importance")
        plt.tight_layout(); st.pyplot(fig)

    with st.expander("Decision tree diagram"):
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(b["tree"], feature_names=b["names"], class_names=list(b["tree"].classes_),
                  filled=True, rounded=True, fontsize=7, ax=ax)
        st.pyplot(fig)
    with st.expander("Full classification report"):
        st.code(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    st.subheader("Reliability assessment")
    beats = m["acc"] > majority
    st.write(f"- Accuracy {m['acc']:.2f} vs majority-class baseline {majority:.2f} → "
             + ("better than always guessing the majority class." if beats else "**not** better than always guessing the majority class."))
    st.write(f"- Not Completed: precision {fmt(m['prec'])}, recall {fmt(m['rec'])}, F1 {fmt(m['f1'])}.")
    st.write(f"- Missed risky bookings (false negatives): {fn}; false alarms (false positives): {fp}.")
    if pd.isna(m["prec"]) or pd.isna(m["rec"]):
        st.warning("Precision or recall is undefined - reported as NaN rather than a false 0 or 1.")
    elif m["prec"] >= 0.70 and m["rec"] >= 0.70 and beats:
        st.success("VERDICT: acceptable as an ADVISORY aid, still with human review.")
    else:
        st.error("VERDICT: NOT reliable enough for automatic decisions (precision and/or recall below the 0.70 target). "
                 "Use only as a soft advisory flag (e.g. send a reminder), never to reject a booking.")
    st.markdown("""
**Limitation:** the data is a small synthetic sample (the test set has only ~80 rows), so metrics are noisy. Real drivers
(weather, exam weeks, event type, a requester's past cancellation history) are not captured, and the classes are imbalanced.

**Improvement:** collect more months of real data, add behavioural features known at submission, compare with Random
Forest / Gradient Boosting, tune the threshold with cross-validation, and monitor for drift after deployment.
""")


def page_predictor():
    st.header("Risk Predictor (decision support)")
    st.caption("Enter a pending reservation. The model returns a **predicted** probability that it will be cancelled or "
               "become a no-show. This is advisory only - the administrator makes the final decision.")
    b = get_bundle()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        facility = c1.selectbox("Facility type", FACILITIES)
        user = c2.selectbox("User category", USERS)
        c3, c4 = st.columns(2)
        slot = c3.selectbox("Time slot", SLOTS)
        res_date = c4.date_input("Reservation date", value=pd.Timestamp("2026-03-16"))
        c5, c6, c7 = st.columns(3)
        participants = c5.number_input("Participants", min_value=1, max_value=300, value=30, step=1)
        lead = c6.number_input("Lead time (days between request and reservation)", min_value=0, max_value=120, value=10, step=1)
        duration = c7.number_input("Duration (hours)", min_value=1, max_value=8, value=2, step=1)
        submitted = st.form_submit_button("Predict risk")

    if not submitted:
        return

    day_type = "Weekend" if pd.Timestamp(res_date).dayofweek >= 5 else "Weekday"
    row = pd.DataFrame([{"Facility_Type": facility, "User_Category": user, "Time_Slot": slot, "Day_Type": day_type,
                         "Participants": participants, "Lead_Time_Days": lead, "Duration_Hours": duration}])[FEATURES]
    p = float(risk_proba(b["pipe"], row)[0])
    lvl = risk_level(p)

    c1, c2 = st.columns(2)
    c1.metric("Predicted probability of Not Completed", f"{p:.0%}")
    c2.metric("Risk level", lvl)
    {"Low": st.success, "Medium": st.warning, "High": st.error}[lvl](
        {"Low": "Low risk - no special action needed.",
         "Medium": "Medium risk - consider sending a reminder.",
         "High": "High risk - send a reminder, request confirmation, or review manually. Do not auto-reject."}[lvl])

    st.subheader("Why? (decision-path factors)")
    for r in explain_prediction(b, row):
        st.write("- " + r)
    st.caption(f"Day type derived from the date: {day_type}. Model test accuracy "
               f"{accuracy_score(b['y_test'], b['pipe'].predict(b['X_test'])):.2f} - treat as a probability, not a fact.")


def page_dashboard():
    st.header("Facility Administrator Dashboard (Task 11)")
    st.caption("🔵🔴 **OBSERVED** = historical, cleaned data.   🟠 **PREDICTED** = model output on test reservations the model had not seen.")
    df, b = load_clean(), get_bundle()
    t = eda_tables(df)

    scored = b["model_df"].loc[b["X_test"].index,
                               ["Reservation_ID", "Facility_Type", "User_Category", "Lead_Time_Days", "Participants"]].copy()
    scored["Risk_Probability"] = risk_proba(b["pipe"], b["X_test"]).round(2)
    scored["Risk_Level"] = scored["Risk_Probability"].map(risk_level)
    scored = scored.sort_values("Risk_Probability", ascending=False)

    used = df[df["Usage_Status"] == "Used"]["Attendance_Ratio"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total reservations", len(df))
    k2.metric("Not-completed rate", f"{100 * (t['eda'][TARGET] == POSITIVE).mean():.1f}%", f"of {t['n_out']}", delta_color="off")
    k3.metric("Avg attendance ratio", f"{used.mean():.2f}", f"of {int(used.notna().sum())} used", delta_color="off")
    k4.metric("Unknown outcome", int(df[TARGET].isna().sum()))
    k5.metric("🟠 High-risk (predicted)", int((scored["Risk_Level"] == "High").sum()), f"of {len(scored)} test", delta_color="off")

    st.subheader("Observed")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    chart_facility(t, axes[0]); chart_user(t, axes[1])
    axes[2].bar(LEAD_ORDER, t["lead_ct"]["pct_not_completed"], color=RED)
    axes[2].set_title("% not completed by lead-time category", fontsize=10)
    axes[2].set_ylabel("% not completed"); axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("🟠 Predicted")
    left, right = st.columns([1, 2])
    with left:
        rc = scored["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.bar(rc.index, rc.values, color=PRED, hatch="//", edgecolor="white")
        ax.set_title("Predicted risk-level distribution", fontsize=10); ax.set_ylabel("Reservations")
        for i, v in enumerate(rc.values):
            ax.text(i, v + 0.3, int(v), ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig)
    with right:
        levels = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High"])
        st.dataframe(scored[scored["Risk_Level"].isin(levels)].reset_index(drop=True),
                     hide_index=True)

    top = scored.iloc[0]
    st.subheader(f"Why is {top['Reservation_ID']} high risk? (predicted {top['Risk_Probability']:.0%})")
    for r in explain_prediction(b, b["X_test"].loc[[top.name]])[:5]:
        st.write("- " + r)
    st.markdown("**Administrator actions:** `Send reminder` · `Request confirmation` · `Approve anyway` · `Review manually`")
    st.caption("Predictions are probabilities, not facts. They support - never replace - the administrator's decision.")


def page_requirements():
    st.header("Requirements & Use Case Model (Tasks 9-10)")
    df, b = load_clean(), get_bundle()
    t = eda_tables(df)
    ut, lc = t["user_tbl"], t["lead_ct"]
    top_user, hi_l, lo_l = ut.index[0], t["hi_l"], t["lo_l"]
    m = safe_metrics(b["y_test"], b["pipe"].predict(b["X_test"]))
    top3 = list(b["importance"].head(3).index)

    st.subheader("Evidence-based functional requirements")
    fr = pd.DataFrame([
        [f"Part III Q2: '{top_user}' has the highest not-completed rate, {ut.loc[top_user, 'pct_not_completed']:.1f}% "
         f"({int(ut.loc[top_user, 'not_completed'])}/{int(ut.loc[top_user, 'n'])}).",
         "FR-DM-01",
         f"The system shall send an automated reminder to requesters in the '{top_user}' category before their reserved "
         "date and shall require them to confirm or cancel the reservation."],
        [f"Part III Q3: '{hi_l}' bookings have the highest not-completed rate, {lc.loc[hi_l, 'pct_not_completed']:.1f}% "
         f"({int(lc.loc[hi_l, POSITIVE])}/{int(lc.loc[hi_l, 'n'])}), vs {lc.loc[lo_l, 'pct_not_completed']:.1f}% for '{lo_l}'.",
         "FR-DM-02",
         f"The system shall prompt requesters of '{hi_l}' reservations to re-confirm before the reserved date and shall "
         "notify the facility administrator of reservations that remain unconfirmed."],
        [f"Part IV: Decision Tree (test n = {len(b['y_test'])}) - recall {fmt(m['rec'])}, precision {fmt(m['prec'])} "
         f"for Not Completed; top predictors: {', '.join(top3)}.",
         "FR-DM-03",
         "The system shall display to the facility administrator a predicted cancellation/no-show risk level (Low, Medium, "
         "High) for each pending reservation, labelled as a prediction, with the main factors behind it, without "
         "automatically approving or rejecting the reservation."],
    ], columns=["Finding / Evidence", "Requirement ID", "Functional Requirement"])
    st.dataframe(fr, hide_index=True)

    st.subheader("Updated UML Use Case Diagram")
    if DIAGRAM_PNG.exists():
        st.image(str(DIAGRAM_PNG), caption="Blue = existing use cases, green = new analytical use cases")
    else:
        st.warning("Use_Case_Diagram.png was not found in the repository.")
    st.markdown("""
- **System boundary:** Campus Facility Reservation and Monitoring System.
- **Actors:** Requester, Facility Administrator, Data Analyst.
- **New analytical use cases:** View Reservation Analytics Dashboard, Predict Cancellation Risk, View Risk Explanation,
  Retrain and Evaluate Risk Model, Clean and Prepare Training Data.
- *Predict Cancellation Risk* **«extend»** *Review and Approve Reservation* (optional); *View Risk Explanation*
  **«extend»** *Predict Cancellation Risk*; *Retrain and Evaluate Risk Model* **«include»** *Clean and Prepare Training Data*
  (mandatory); *Submit Reservation Request* **«include»** *Validate Reservation Data*.
""")


def page_critical():
    st.header("Critical Analysis (Task 12)")
    with st.expander("(a) Why should predicted high risk not automatically reject a reservation?", expanded=True):
        st.write("A prediction is a probability based on past patterns, not a fact about this particular booking. The model "
                 "makes errors - false positives would unfairly deny people who would have used the facility - it only sees a "
                 "few variables, and group-level patterns (e.g. a user category) must not become discrimination against "
                 "individuals. Automatic rejection would also create unused capacity and unfairness with no appeal. A "
                 "high-risk flag should trigger softer actions - a reminder, a confirmation request, or administrator "
                 "review - and the human administrator makes the final decision.")
    with st.expander("(b) How does poor data quality affect the model and the information system?", expanded=True):
        st.write("Duplicates inflate counts, inconsistent labels (`Canceled` vs `Cancelled`) split one category into several, "
                 "invalid values (999 participants, negative lead time) distort statistics, and missing outcomes make labels "
                 "unknowable. The model then learns wrong patterns (garbage in, garbage out), accuracy figures become "
                 "misleading, and administrators lose trust in dashboards. For the information system itself, bad data shows "
                 "that input validation and controlled vocabularies (drop-down lists) are missing at data entry, so quality "
                 "must be enforced at the source, not only cleaned later.")
    with st.expander("(c) What information should explain a high-risk prediction?", expanded=True):
        st.write("(1) The risk level and probability, clearly labelled *Predicted*; (2) the top contributing factors in plain "
                 "words (user category, lead time, facility type, weekend booking); (3) historical evidence for those factors "
                 "with denominators (\"x of n similar bookings were not completed\"); (4) the model's reliability (test "
                 "accuracy/recall and date of last training); (5) any missing or uncertain inputs; and (6) suggested actions "
                 "with the final approve/reject choice left to the administrator.")


# --------------------------------------------------------------------------- router
PAGES = {
    "Overview": page_overview,
    "Data & Cleaning": page_data,
    "Exploratory Analysis": page_eda,
    "Model & Reliability": page_model,
    "Risk Predictor": page_predictor,
    "Admin Dashboard": page_dashboard,
    "Requirements & Use Case": page_requirements,
    "Critical Analysis": page_critical,
}


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("April Jhon B. Relampago  \nBSIS 3A  \nSystems Analysis and Design")
    PAGES[choice]()


main()
