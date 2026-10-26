# # Laboratory Examination — Data Mining and Systems Analysis Integration
# **Course:** Systems Analysis and Design (SAD) | **Program:** BS Information Systems
# **Name:** April Jhon B. Relampago | **Section:** BSIS 3A | **Date:** 10/12/2026
#
# **System studied:** Campus Facility Reservation and Monitoring System (prototype)
#
# ### How to run
# 1. Open in Google Colab → **Runtime → Run all**.
# 2. **No CSV upload is needed.** Section 0 *generates* a synthetic `facility_reservations.csv` (with intentional missing values, duplicates, inconsistent categories and invalid values) and saves it in the Colab working folder.
# 3. The last cell saves/downloads `facility_reservations.csv` (raw) and `facility_reservations_clean.csv` (cleaned).
#
# **Pipeline:** Historical data → cleaning → analysis → prediction → evidence-based requirements → use cases → decision support.
#
# | Part | Content |
# |---|---|
# | 0 | Synthetic dataset generator (instructor preparation) |
# | I | Dataset acquisition and understanding (Tasks 1–3) |
# | II | Data cleaning and feature engineering (Tasks 4–5) |
# | III | Exploratory data analysis (Task 6) |
# | IV | Decision Tree modelling and reliability (Tasks 7–8) |
# | V | Requirements analysis and use case modelling (Tasks 9–10) |
# | VI | Decision-support dashboard and critical analysis (Tasks 11–12) |

# ---- Setup -----------------------------------------------------------
import re, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score, classification_report,
                             make_scorer)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

RANDOM_STATE = 42                      # fixed seed -> reproducible data, split and model
RAW_CSV   = "facility_reservations.csv"
CLEAN_CSV = "facility_reservations_clean.csv"
print("Setup complete. Working folder:", os.getcwd())

# ---
# ## Section 0 — Synthetic Dataset Generator (no upload required)
# Per the *Instructor Preparation Notes*, this cell creates **≈ 420 synthetic reservation transactions** and deliberately injects a manageable number of data-quality problems:
#
# | Problem type | What is injected |
# |---|---|
# | Missing values | `Facility_Type`, `User_Category`, `Participants`, `Duration_Hours`, `Usage_Status`, `Lead_Time_Days`, `Request_Date`, and `Actual_Attendance` for some *Used* records |
# | Duplicates | exact duplicate rows + near-duplicates that differ only by letter-case |
# | Inconsistent categories | `Classroom / classroom / CLASSROOM / Class Room`, `No-Show / no show / NoShow`, etc. |
# | Invalid numbers | Participants = 0, negative or 999; negative lead time; negative attendance; duration 0 or 24 h; reservation date before request date |
#
# All data is **synthetic** — no real student information is used.

def generate_facility_reservations(n=420, seed=RANDOM_STATE):
    """Create a clean synthetic dataset, then inject intentional data-quality problems."""
    rng = np.random.default_rng(seed)

    facility_types = ["Classroom", "Computer Lab", "Auditorium", "Sports Court", "Conference Room", "Study Room"]
    fac_p  = np.array([0.28, 0.17, 0.09, 0.15, 0.12, 0.19]); fac_p /= fac_p.sum()
    user_cats = ["Student", "Student Organization", "Faculty", "Staff", "External Guest"]
    user_p = np.array([0.38, 0.22, 0.22, 0.12, 0.06]);       user_p /= user_p.sum()

    facility = rng.choice(facility_types, n, p=fac_p)
    user     = rng.choice(user_cats, n, p=user_p)
    slot     = rng.choice(["Morning", "Afternoon", "Evening"], n, p=[0.45, 0.40, 0.15])
    duration = rng.choice([1, 2, 3, 4], n, p=[0.30, 0.40, 0.20, 0.10])

    size = {"Classroom": (35, 8), "Computer Lab": (30, 6), "Auditorium": (150, 50),
            "Sports Court": (25, 8), "Conference Room": (12, 4), "Study Room": (5, 2)}
    participants = np.array([int(np.clip(round(rng.normal(*size[f])), 2, 300)) for f in facility])

    # --- lead time (days between request and reserved date) ---
    grp  = rng.choice(4, n, p=[0.18, 0.30, 0.27, 0.25])
    lead = np.where(grp == 0, rng.integers(0, 3, n),
           np.where(grp == 1, rng.integers(3, 8, n),
           np.where(grp == 2, rng.integers(8, 15, n), rng.integers(15, 46, n))))
    res_dates = pd.Timestamp("2026-02-02") + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    req_dates = res_dates - pd.to_timedelta(lead, unit="D")
    weekend   = np.asarray(res_dates.dayofweek >= 5)

    # --- outcome model (hidden "ground truth" pattern the Decision Tree must discover) ---
    lead_eff = np.select([lead <= 2, lead <= 7, lead <= 14], [-0.4, 0.0, 0.3], default=0.9)
    user_eff = pd.Series(user).map({"Student": 0.3, "Student Organization": 0.7, "Faculty": -0.5,
                                    "Staff": -0.3, "External Guest": 0.5}).to_numpy()
    fac_eff  = pd.Series(facility).map({"Classroom": -0.2, "Computer Lab": -0.1, "Auditorium": 0.2,
                                        "Sports Court": 0.5, "Conference Room": -0.3, "Study Room": 0.3}).to_numpy()
    logit = -1.5 + 1.6 * (lead_eff + user_eff + fac_eff + 0.4 * weekend + np.where(slot == "Evening", 0.3, 0.0))
    p_not = 1 / (1 + np.exp(-logit))
    not_completed = rng.random(n) < p_not
    cancelled = not_completed & (rng.random(n) < 0.60)
    noshow    = not_completed & ~cancelled
    status = np.where(cancelled, "Cancelled", np.where(noshow, "No-Show", "Used"))

    cancel_days = np.where(cancelled, rng.integers(0, lead + 1), np.nan)
    attendance  = np.where(status == "Used", np.round(participants * rng.uniform(0.55, 1.05, n)),
                  np.where(status == "No-Show", 0, np.nan))

    df = pd.DataFrame({
        "Reservation_ID": [f"RSV-{i:04d}" for i in range(1, n + 1)],
        "Facility_Type": facility,
        "User_Category": user,
        "Request_Date": req_dates.strftime("%Y-%m-%d"),
        "Reservation_Date": res_dates.strftime("%Y-%m-%d"),
        "Lead_Time_Days": lead,
        "Time_Slot": slot,
        "Duration_Hours": duration,
        "Participants": participants,
        "Usage_Status": status,
        "Actual_Attendance": attendance,
        "Cancellation_Days_Before": cancel_days,
    })

    # ------------------- inject intentional data-quality problems -------------------
    pool = list(rng.permutation(len(df)))
    def take(k):
        idx = pool[:k]; del pool[:k]; return idx

    # missing values
    for col, k in {"Facility_Type": 6, "User_Category": 5, "Participants": 8, "Duration_Hours": 5,
                   "Usage_Status": 5, "Lead_Time_Days": 6, "Request_Date": 3}.items():
        df.loc[take(k), col] = np.nan
    used_idx = df.index[df["Usage_Status"] == "Used"]
    df.loc[rng.choice(used_idx, 4, replace=False), "Actual_Attendance"] = np.nan   # Used but attendance not recorded

    # invalid numeric / date values
    df.loc[take(4), "Participants"] = 0
    df.loc[take(3), "Participants"] = -5
    df.loc[take(3), "Participants"] = 999
    df.loc[take(5), "Lead_Time_Days"] = -3
    df.loc[take(3), "Actual_Attendance"] = -2
    df.loc[take(3), "Duration_Hours"] = 0
    df.loc[take(2), "Duration_Hours"] = 24
    for i in take(4):                                                   # reservation date BEFORE request date
        df.loc[i, ["Request_Date", "Reservation_Date"]] = df.loc[i, ["Reservation_Date", "Request_Date"]].values

    # inconsistent category spellings
    fac_alias  = {"Classroom": "Class Room", "Computer Lab": "Comp Lab", "Auditorium": "Auditorium  ",
                  "Sports Court": "Sport Court", "Conference Room": "Conf. Room", "Study Room": "Study room"}
    user_alias = {"Student": "Stud.", "Student Organization": "Student Org", "Faculty": "Faculty Member",
                  "Staff": "Staff ", "External Guest": "External"}
    stat_alias = {"Used": "Used ", "Cancelled": "Canceled", "No-Show": "No Show"}
    def messy(v, alias):
        r = int(rng.integers(0, 4))
        return [v.lower(), v.upper(), " " + v + " ", alias.get(v, v)][r]
    for col, alias, k in [("Facility_Type", fac_alias, 25), ("User_Category", user_alias, 20), ("Usage_Status", stat_alias, 25)]:
        for i in take(k):
            if pd.notna(df.at[i, col]):
                df.at[i, col] = messy(df.at[i, col], alias)

    # duplicates: exact copies + near-duplicates (differ only by letter-case)
    exact = df.sample(14, random_state=seed)
    near  = df.sample(3, random_state=seed + 1).copy()
    near["User_Category"] = near["User_Category"].str.upper()
    df = pd.concat([df, exact, near], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

raw_generated = generate_facility_reservations()
raw_generated.to_csv(RAW_CSV, index=False)
print(f"Generated '{RAW_CSV}' with {raw_generated.shape[0]} rows x {raw_generated.shape[1]} columns "
      f"(saved in {os.getcwd()}).")

# ---
# # PART I — Dataset Acquisition and Understanding (15 pts)
# ### Task 1 — Load the dataset, preview, shape, data types, summary statistics

# Load the CSV with pandas (the file was just generated in Section 0)
df_raw = pd.read_csv(RAW_CSV)

print("=== First 10 records ===")
display(df_raw.head(10))

print(f"\nNumber of rows: {df_raw.shape[0]}   |   Number of columns: {df_raw.shape[1]}")

print("\n=== Variable types ===")
display(df_raw.dtypes.rename("dtype").to_frame())

print("\n=== Summary statistics (numerical) ===")
display(df_raw.describe())

print("\n=== Summary statistics (categorical / text) ===")
display(df_raw.select_dtypes(exclude="number").describe())

# > **Observation (raw data):** the numeric summary already exposes problems — e.g. `Participants` has a **negative minimum and a maximum of 999**, `Lead_Time_Days` has a **negative minimum**, `Duration_Hours` reaches **24**, and `Actual_Attendance` has a negative value. Counts also differ between columns, which signals **missing values**. These are handled in Part II.

# ### Task 2 — Predictors, target, categorical and numerical variables
#
# | Role | Variable | Type | Why it matters |
# |---|---|---|---|
# | **Predictor 1** | `Facility_Type` | Categorical | Different facilities (e.g. sports court vs. classroom) have different usage patterns. Known when the request is submitted. |
# | **Predictor 2** | `User_Category` | Categorical | Student / Student Org / Faculty / Staff / External Guest behave differently in committing to bookings. |
# | **Predictor 3** | `Participants` | Numerical | Declared group size; larger events are riskier to coordinate. |
# | **Predictor 4** | `Lead_Time_Days` | Numerical | Days between request and reserved date; long gaps give more time for plans to change. |
# | **Predictor 5** | `Duration_Hours` | Numerical | Requested length of use. (`Time_Slot` and weekday/weekend, derived from the reservation date, are extra predictors.) |
# | **Target** | `Reservation_Outcome` | Categorical (binary) | Derived from `Usage_Status`: **Completed** vs **Not Completed**. This is what the model predicts. |
# | **2 categorical** | `Facility_Type`, `User_Category` | | Encoded with one-hot encoding before modelling. |
# | **2 numerical** | `Participants`, `Lead_Time_Days` | | Used directly (after imputing missing values with the training median). |
#
# **Not predictors (identifier or post-outcome):** `Reservation_ID` (identifier only); `Usage_Status`, `Actual_Attendance`, `Cancellation_Days_Before` (only known **after** the outcome → *data leakage*).

# ---- Brief data dictionary ----------------------------------------------------------
data_dictionary = pd.DataFrame([
    ["Reservation_ID",          "Identifier",   "Unique reservation transaction code (e.g. RSV-0001)",                    "Identifier (not used)"],
    ["Facility_Type",           "Categorical",  "Type of facility reserved (Classroom, Computer Lab, Auditorium, ...)",   "Predictor"],
    ["User_Category",           "Categorical",  "Who made the request (Student, Student Organization, Faculty, ...)",     "Predictor"],
    ["Request_Date",            "Date",         "Date the reservation request was submitted",                             "Used to derive lead time"],
    ["Reservation_Date",        "Date",         "Date the facility is reserved for",                                      "Used to derive weekday/weekend"],
    ["Lead_Time_Days",          "Numerical",    "Days between Request_Date and Reservation_Date",                          "Predictor"],
    ["Time_Slot",               "Categorical",  "Morning / Afternoon / Evening",                                          "Predictor"],
    ["Duration_Hours",          "Numerical",    "Requested duration of use in hours",                                     "Predictor"],
    ["Participants",            "Numerical",    "Declared number of participants at submission",                          "Predictor"],
    ["Usage_Status",            "Categorical",  "Used / Cancelled / No-Show (known only AFTER the event)",                "Source of target - NOT a predictor"],
    ["Actual_Attendance",       "Numerical",    "Persons who actually attended (known only after the event)",             "Post-outcome - NOT a predictor"],
    ["Cancellation_Days_Before","Numerical",    "Days before the event that a cancellation was made",                     "Post-outcome - NOT a predictor"],
], columns=["Variable", "Type", "Description", "Role"])
display(data_dictionary)

# ### Task 3 — Obtaining the same data from the university reservation system
# **Option A – Database query (SQL).** An analyst with read-only access can extract the data directly from the reservation database:
# ```sql
# SELECT r.reservation_id, f.facility_type, u.user_category,
#        r.request_date, r.reservation_date,
#        DATEDIFF(r.reservation_date, r.request_date) AS lead_time_days,
#        r.time_slot, r.duration_hours, r.participants,
#        r.usage_status, r.actual_attendance, r.cancellation_days_before
# FROM   reservations r
# JOIN   facilities f ON f.facility_id = r.facility_id
# JOIN   users      u ON u.user_id     = r.user_id
# WHERE  r.reservation_date BETWEEN '2026-02-01' AND '2026-06-30';
# ```
# **Option B – REST API.** The system could expose an authenticated endpoint such as `GET /api/v1/reservations?from=2026-02-01&to=2026-06-30&format=json`. Python would call it with `requests.get(url, headers={"Authorization": "Bearer <token>"})`, convert the JSON with `pd.DataFrame(response.json())`, and save it as CSV.
#
# **Validation check during collection (implemented below):** *a reservation date must not be earlier than its request date, and the stored lead time must equal that date difference.* Records that fail are flagged at collection time instead of silently entering the analysis.

# Example collection-time validation check on the raw extract
req = pd.to_datetime(df_raw["Request_Date"],     errors="coerce")
res = pd.to_datetime(df_raw["Reservation_Date"], errors="coerce")
computed_lead = (res - req).dt.days

fails_order    = (computed_lead < 0)
fails_mismatch = computed_lead.notna() & df_raw["Lead_Time_Days"].notna() & (computed_lead >= 0) & (computed_lead != df_raw["Lead_Time_Days"])
print("Validation: Reservation_Date must be >= Request_Date, and stored lead time must match the date difference")
print(f"  Records where reservation date precedes request date : {int(fails_order.sum())}")
print(f"  Records where stored lead time != computed difference: {int(fails_mismatch.sum())}")
display(df_raw.loc[fails_order | fails_mismatch,
                   ["Reservation_ID", "Request_Date", "Reservation_Date", "Lead_Time_Days"]].head(10))

# ---
# # PART II — Data Cleaning and Feature Engineering (20 pts)
# ### Task 4 — Detect and address data-quality problems
# **Principle:** *never invent values the record does not support.* The original dataset is preserved as `df_raw`; all cleaning is done on a copy (`df`).
#
# | Problem | Decision and justification |
# |---|---|
# | **Inconsistent categories** | Trim spaces, ignore case/punctuation, then map every variant to one official label (e.g. `Class Room`, `CLASSROOM` → `Classroom`; `No Show`, `no-show` → `No-Show`; `Canceled` → `Cancelled`). This is a *recoding*, not an invention. |
# | **Duplicate transactions** | Remove duplicates **after** standardising (near-duplicates only differ by case). The same reservation must not be counted twice. |
# | **Missing `Facility_Type` / `User_Category`** | Cannot be known from the record → labelled `Unknown` (kept visible, not guessed). |
# | **Missing `Usage_Status`** | Outcome cannot be determined → `Reservation_Outcome` stays **missing**; those rows are excluded from outcome analysis and modelling (not guessed). |
# | **Invalid `Participants`** (≤ 0 or > 300, the largest facility capacity) | Set to missing (`NaN`). Impossible values are not corrected by guessing. |
# | **Invalid `Duration_Hours`** (< 1 or > 8) | Set to missing. |
# | **Invalid `Actual_Attendance`** (negative) | Set to missing. |
# | **Lead time** | Recomputed from the two dates when both are valid and in order. If a date is missing, the stored value is used if it is ≥ 0. If dates conflict (reservation before request), lead time is set to missing. |
# | **Missing numerics** | Left as `NaN` in the clean CSV. Imputation (training-set median) happens **inside the model pipeline** only, so no test information leaks into training. |

# ---------------- Task 4: cleaning (original preserved in df_raw) ----------------
df = df_raw.copy()                          # working copy; df_raw stays untouched
before = {"rows": len(df_raw), "exact_duplicates": int(df_raw.duplicated().sum())}

# 1) Inconsistent categories -> standardise
def norm_key(s):
    return s.astype(object).where(s.notna(), np.nan).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

FAC_MAP  = {"classroom": "Classroom", "computerlab": "Computer Lab", "complab": "Computer Lab",
            "auditorium": "Auditorium", "sportscourt": "Sports Court", "sportcourt": "Sports Court",
            "conferenceroom": "Conference Room", "confroom": "Conference Room", "studyroom": "Study Room"}
USER_MAP = {"student": "Student", "stud": "Student", "studentorganization": "Student Organization",
            "studentorg": "Student Organization", "faculty": "Faculty", "facultymember": "Faculty",
            "staff": "Staff", "externalguest": "External Guest", "external": "External Guest"}
STAT_MAP = {"used": "Used", "cancelled": "Cancelled", "canceled": "Cancelled", "noshow": "No-Show"}
SLOT_MAP = {"morning": "Morning", "afternoon": "Afternoon", "evening": "Evening"}

variants_before = {c: df_raw[c].nunique() for c in ["Facility_Type", "User_Category", "Usage_Status"]}
for col, mp in [("Facility_Type", FAC_MAP), ("User_Category", USER_MAP), ("Usage_Status", STAT_MAP), ("Time_Slot", SLOT_MAP)]:
    key = norm_key(df[col])
    out = key.map(mp)
    unmapped = df.loc[out.isna() & df[col].notna(), col].unique()
    assert len(unmapped) == 0, f"Unmapped values in {col}: {unmapped}"
    df[col] = out
variants_after = {c: df[c].nunique() for c in ["Facility_Type", "User_Category", "Usage_Status"]}

# 2) Missing categories -> explicit "Unknown" (Usage_Status is NOT filled)
for c in ["Facility_Type", "User_Category"]:
    df[c] = df[c].fillna("Unknown")

# 3) Duplicates (after standardising, so near-duplicates are detected)
n_before_dedup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
dup_removed = n_before_dedup - len(df)
id_dups_left = int(df["Reservation_ID"].duplicated().sum())

# 4) Invalid numerical values -> NaN (not invented)
MAX_PARTICIPANTS = 300
bad_part = (df["Participants"] <= 0) | (df["Participants"] > MAX_PARTICIPANTS)
bad_dur  = (df["Duration_Hours"] < 1) | (df["Duration_Hours"] > 8)
bad_att  = df["Actual_Attendance"] < 0
n_bad = {"Participants": int(bad_part.sum()), "Duration_Hours": int(bad_dur.sum()), "Actual_Attendance": int(bad_att.sum())}
df.loc[bad_part, "Participants"] = np.nan
df.loc[bad_dur,  "Duration_Hours"] = np.nan
df.loc[bad_att,  "Actual_Attendance"] = np.nan

# 5) Dates and lead time (supported by the record only)
df["Request_Date"]     = pd.to_datetime(df["Request_Date"],     errors="coerce")
df["Reservation_Date"] = pd.to_datetime(df["Reservation_Date"], errors="coerce")
computed = (df["Reservation_Date"] - df["Request_Date"]).dt.days
reported = df["Lead_Time_Days"].where(df["Lead_Time_Days"] >= 0)
neg_reported = int((df["Lead_Time_Days"] < 0).sum())
conflict_dates = int((computed < 0).sum())
df["Lead_Time_Days"] = np.where(computed >= 0, computed, np.where(computed.isna(), reported, np.nan))

# 6) Logical consistency check: cancelled bookings should have no attendance
cancelled_with_att = int(((df["Usage_Status"] == "Cancelled") & df["Actual_Attendance"].notna()).sum())

print("Cleaning log")
print(f"  Category variants  (before -> after): {variants_before} -> {variants_after}")
print(f"  Duplicate rows removed               : {dup_removed}  (remaining duplicate Reservation_IDs: {id_dups_left})")
print(f"  Invalid values set to NaN            : {n_bad}")
print(f"  Negative lead times in raw data      : {neg_reported}  | reservation date before request date: {conflict_dates}")
print(f"  Cancelled rows that have attendance  : {cancelled_with_att}")

# ### Task 5 — Derived variables
# | Variable | Rule applied |
# |---|---|
# | `Lead_Time_Category` | 0–2 days = **Last Minute**; 3–7 = **Short Lead**; 8–14 = **Medium Lead**; > 14 = **Early Booking**; unknown if lead time is missing |
# | `Reservation_Outcome` | **Completed** if `Usage_Status = Used`; **Not Completed** if `Cancelled` or `No-Show`; left missing if status is missing |
# | `Attendance_Ratio` | `Actual_Attendance / Participants`; **NaN** (not 0, not guessed) when attendance is missing, participants is missing, or participants = 0 |
#
# Additional helper features (known at submission): `Reservation_Weekday` and `Day_Type` (Weekday / Weekend) from the reservation date.

# ---------------- Task 5: feature engineering ----------------
# Lead_Time_Category
bins   = [-0.1, 2, 7, 14, np.inf]
labels = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
df["Lead_Time_Category"] = pd.cut(df["Lead_Time_Days"], bins=bins, labels=labels).astype(object)
df["Lead_Time_Category"] = df["Lead_Time_Category"].where(df["Lead_Time_Days"].notna(), "Unknown")

# Reservation_Outcome (missing stays missing)
df["Reservation_Outcome"] = df["Usage_Status"].map({"Used": "Completed", "Cancelled": "Not Completed", "No-Show": "Not Completed"})

# Attendance_Ratio: safe division (zero / missing denominator -> NaN)
safe_den = df["Participants"].where(df["Participants"] > 0)          # 0 or NaN participants -> NaN
df["Attendance_Ratio"] = (df["Actual_Attendance"] / safe_den).round(3)

# Helper features available at submission time
df["Reservation_Weekday"] = df["Reservation_Date"].dt.day_name()
df["Day_Type"] = pd.Series(np.where(df["Reservation_Date"].dt.dayofweek >= 5, "Weekend", "Weekday"),
                           index=df.index, dtype=object).where(df["Reservation_Date"].notna())

display(df[["Reservation_ID", "Lead_Time_Days", "Lead_Time_Category", "Usage_Status", "Reservation_Outcome",
            "Participants", "Actual_Attendance", "Attendance_Ratio"]].head(10))
print("Lead_Time_Category counts:\n", df["Lead_Time_Category"].value_counts(), "\n")
print("Reservation_Outcome counts (incl. missing):\n", df["Reservation_Outcome"].value_counts(dropna=False))

# ---------------- Data-quality summary + clean CSV ----------------
dq_rows = pd.DataFrame({
    "Raw_missing": df_raw.isna().sum(),
    "Clean_missing_NaN": df.reindex(columns=df_raw.columns).isna().sum(),
})
dq_rows["Clean_labelled_Unknown"] = [int((df[c] == "Unknown").sum()) if c in ("Facility_Type", "User_Category") else 0
                                    for c in dq_rows.index]
print("=== Missing values per column: raw vs clean ===")
print("(Clean NaN can exceed raw NaN because invalid values were deliberately converted to NaN; duplicates were also removed.)")
display(dq_rows)

summary = pd.Series({
    "Raw rows": before["rows"],
    "Exact duplicate rows in raw file": before["exact_duplicates"],
    "Duplicates removed after standardising": dup_removed,
    "Clean rows": len(df),
    "Invalid Participants set to NaN": n_bad["Participants"],
    "Invalid Duration_Hours set to NaN": n_bad["Duration_Hours"],
    "Invalid Actual_Attendance set to NaN": n_bad["Actual_Attendance"],
    "Rows with unknown Reservation_Outcome (excluded from modelling)": int(df["Reservation_Outcome"].isna().sum()),
    "Rows with Attendance_Ratio not computable": int(df["Attendance_Ratio"].isna().sum()),
}, name="Value")
print("\n=== Data-quality summary ===")
display(summary.to_frame())

df_out = df.copy()
for c in ["Request_Date", "Reservation_Date"]:
    df_out[c] = df_out[c].dt.strftime("%Y-%m-%d")
df_out.to_csv(CLEAN_CSV, index=False)
print(f"\nSaved cleaned dataset -> {CLEAN_CSV}  ({df_out.shape[0]} rows x {df_out.shape[1]} columns)")

# ---
# # PART III — Exploratory Data Analysis (15 pts)
# ### Task 6 — Business questions
# Percentages always state their **denominator**. Rows with `Unknown` category or a missing outcome are excluded from the relevant question (they cannot be assigned to a group), and the exclusion is reported.

# ---------- Q1: facility type with the highest reservation count ----------
known_fac = df[df["Facility_Type"] != "Unknown"]
fac_counts = known_fac["Facility_Type"].value_counts()
N_FAC = int(fac_counts.sum())
top_fac, top_fac_n = fac_counts.index[0], int(fac_counts.iloc[0])

# ---------- Q2: user category with highest cancellation / no-show % ----------
eda = df[df["Reservation_Outcome"].notna()].copy()
N_OUT = len(eda)
user_tbl = (eda[eda["User_Category"] != "Unknown"]
            .groupby("User_Category")["Reservation_Outcome"]
            .agg(n="size", not_completed=lambda s: int((s == "Not Completed").sum())))
user_tbl["pct_not_completed"] = (100 * user_tbl["not_completed"] / user_tbl["n"]).round(1)
user_tbl = user_tbl.sort_values("pct_not_completed", ascending=False)
top_user = user_tbl.index[0]

# ---------- Q3: lead-time category vs outcome ----------
order = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
lead_eda = eda[eda["Lead_Time_Category"].isin(order)]
lead_ct  = pd.crosstab(lead_eda["Lead_Time_Category"], lead_eda["Reservation_Outcome"]).reindex(order)
lead_ct["n"] = lead_ct.sum(axis=1)
lead_ct["pct_not_completed"] = (100 * lead_ct["Not Completed"] / lead_ct["n"]).round(1)

print("Q1 table - reservations per facility type (denominator = {} reservations with known facility)".format(N_FAC))
display(fac_counts.rename("reservations").to_frame().assign(share_pct=(100 * fac_counts / N_FAC).round(1)))
print("Q2 table - Not Completed (cancelled + no-show) by user category (denominator per row = n)")
display(user_tbl)
print("Q3 table - outcome by lead-time category (denominator per row = n)")
display(lead_ct)

# ---------------- Visualisations ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# Chart 1: reservation count by facility type
ax = axes[0]
bars = ax.bar(fac_counts.index, fac_counts.values, color="#4C78A8")
ax.set_title(f"Reservations by Facility Type\n(n = {N_FAC} reservations with known facility)")
ax.set_ylabel("Number of reservations"); ax.set_xlabel("Facility type")
ax.tick_params(axis="x", rotation=30)
for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, int(b.get_height()), ha="center", fontsize=9)

# Chart 2: % not completed by user category
ax = axes[1]
bars = ax.bar(user_tbl.index, user_tbl["pct_not_completed"], color="#E45756")
ax.set_title(f"Cancellation / No-Show Rate by User Category\n(denominator = reservations with known outcome per category)")
ax.set_ylabel("% Not Completed"); ax.set_xlabel("User category"); ax.tick_params(axis="x", rotation=30)
ax.set_ylim(0, max(60, user_tbl["pct_not_completed"].max() + 12))
for b, (n, k, p) in zip(bars, user_tbl[["n", "not_completed", "pct_not_completed"]].values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{p:.1f}%\n({int(k)}/{int(n)})", ha="center", fontsize=8)

# Chart 3: lead time category vs outcome (100% stacked)
ax = axes[2]
pct_c = 100 * lead_ct["Completed"] / lead_ct["n"]
pct_n = 100 * lead_ct["Not Completed"] / lead_ct["n"]
ax.bar(order, pct_c, color="#59A14F", label="Completed")
ax.bar(order, pct_n, bottom=pct_c, color="#E45756", label="Not Completed")
ax.set_title("Outcome by Lead-Time Category\n(100% stacked; n shown under each label)")
ax.set_ylabel("% of reservations in category"); ax.set_xlabel("Lead-time category")
ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{o}\n(n={int(lead_ct.loc[o,'n'])})" for o in order])
for i, o in enumerate(order):
    ax.text(i, pct_c[o] + pct_n[o]/2, f"{pct_n[o]:.1f}%", ha="center", color="white", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
plt.tight_layout(); plt.show()

# ---------------- Evidence-based interpretation (auto-written from the data) ----------------
low_u, high_u = user_tbl.iloc[-1], user_tbl.iloc[0]
hi_l = lead_ct["pct_not_completed"].idxmax(); lo_l = lead_ct["pct_not_completed"].idxmin()
print("FINDINGS")
print(f"Q1. '{top_fac}' has the highest reservation count: {top_fac_n} of {N_FAC} reservations with a known facility type "
      f"({100*top_fac_n/N_FAC:.1f}%). {len(df)-N_FAC} reservations with 'Unknown' facility were excluded.")
print(f"Q2. '{top_user}' has the highest cancellation/no-show rate: {int(high_u['not_completed'])} of {int(high_u['n'])} "
      f"= {high_u['pct_not_completed']:.1f}%, versus {low_u['pct_not_completed']:.1f}% for '{user_tbl.index[-1]}' "
      f"({int(low_u['not_completed'])}/{int(low_u['n'])}). Denominator: reservations with known outcome and known category "
      f"({int(user_tbl['n'].sum())} of {N_OUT}).")
if (user_tbl["n"] < 30).any():
    small = ", ".join(user_tbl.index[user_tbl["n"] < 30])
    print(f"    Caution: small groups (n < 30) - {small} - give less stable percentages.")
print(f"Q3. Not-completed rate is highest for '{hi_l}' ({lead_ct.loc[hi_l,'pct_not_completed']:.1f}% of {int(lead_ct.loc[hi_l,'n'])}) "
      f"and lowest for '{lo_l}' ({lead_ct.loc[lo_l,'pct_not_completed']:.1f}% of {int(lead_ct.loc[lo_l,'n'])}). "
      f"Rates by band: " + "; ".join(f"{o} {lead_ct.loc[o,'pct_not_completed']:.1f}%" for o in order) + ".")
print("Note: these are associations in this dataset, not proof of causation.")

# **How to read the charts (write-up for the report):**
# * **Chart 1 – Reservations by facility type:** a simple count bar chart showing which facilities carry the most demand; the tallest bar is the answer to Q1 and indicates where capacity pressure and monitoring effort should concentrate.
# * **Chart 2 – Cancellation/No-Show rate by user category:** each bar is `Not Completed ÷ reservations with known outcome` for that group (the counts are printed above the bars). The tallest bar answers Q2; groups with small *n* should be interpreted cautiously.
# * **Chart 3 – Outcome by lead-time category:** each column adds to 100 %. If the red share rises or falls across the bands, lead time is related to reliability (see the printed Q3 finding above for the exact direction and numbers).

# ---
# # PART IV — Data Mining and Predictive Modelling (25 pts)
# ### Task 7 — Decision Tree classifier for `Reservation_Outcome`
# **Predictors (all available when the reservation is submitted):** `Facility_Type`, `User_Category`, `Time_Slot`, `Day_Type` (categorical, one-hot encoded) and `Participants`, `Lead_Time_Days`, `Duration_Hours` (numerical) → **7 predictors**.
#
# **How data leakage was avoided**
# 1. The feature list is built explicitly — a target-derived or post-outcome column cannot enter by accident.
# 2. `Usage_Status` (the source of the target), `Actual_Attendance`, `Attendance_Ratio` and `Cancellation_Days_Before` are only known *after* the reservation is used, cancelled, or missed, so they are excluded; an `assert` below enforces this.
# 3. `Reservation_ID` is an identifier, not a predictor.
# 4. Missing-value imputation and encoding live **inside a scikit-learn Pipeline** fitted on the training set only, so no test-set information influences training.
# 5. A **fixed `random_state = 42`** is used for the split and the tree; the split is **stratified** to keep the class ratio in both sets.

# ---------------- Task 7: build the model ----------------
CAT_FEATURES = ["Facility_Type", "User_Category", "Time_Slot", "Day_Type"]
NUM_FEATURES = ["Participants", "Lead_Time_Days", "Duration_Hours"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "Reservation_Outcome"
POSITIVE = "Not Completed"            # the "risk" class administrators care about
LABELS = ["Completed", "Not Completed"]

# ---- leakage guard ----
FORBIDDEN = {"Usage_Status", "Actual_Attendance", "Attendance_Ratio", "Cancellation_Days_Before", "Reservation_Outcome", "Reservation_ID"}
assert FORBIDDEN.isdisjoint(FEATURES), "Leakage! A post-outcome column is in the predictor list."
print(f"Leakage check passed. {len(FEATURES)} predictors:", FEATURES)

model_df = df[df[TARGET].notna()].reset_index(drop=True)
print(f"Rows used for modelling: {len(model_df)} (excluded {len(df)-len(model_df)} rows with unknown outcome)")
X, y = model_df[FEATURES], model_df[TARGET]
print("Class distribution:\n", y.value_counts().to_string())

# ---- 80/20 split (stratified when possible) ----
can_stratify = y.value_counts().min() >= 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y if can_stratify else None)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
if y_test.nunique() < 2:
    print("WARNING: the test set contains only one class - recall/precision for the missing class are undefined (handled below).")

# ---- preprocessing + model ----
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_FEATURES),
])
pipe = Pipeline([
    ("prep", preprocess),
    ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ---------------- Evaluation ----------------
cm = confusion_matrix(y_test, y_pred, labels=LABELS)
acc = accuracy_score(y_test, y_pred)

def safe_pos_metrics(y_true, y_hat, pos=POSITIVE):
    """Precision/recall/F1 for the positive class; returns NaN (instead of crashing or silently 0) when undefined."""
    n_pos_true, n_pos_pred = int((y_true == pos).sum()), int((y_hat == pos).sum())
    prec = precision_score(y_true, y_hat, pos_label=pos, zero_division=0) if n_pos_pred > 0 else np.nan
    rec  = recall_score(y_true, y_hat, pos_label=pos, zero_division=0)    if n_pos_true > 0 else np.nan
    f1   = f1_score(y_true, y_hat, pos_label=pos, zero_division=0)        if (n_pos_true > 0 and n_pos_pred > 0) else np.nan
    return prec, rec, f1, n_pos_true, n_pos_pred

prec, rec, f1, n_pos_true, n_pos_pred = safe_pos_metrics(y_test, y_pred)
majority_acc = y_test.value_counts(normalize=True).max()

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix (test set)"); plt.tight_layout(); plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"Confusion matrix counts (rows = actual, cols = predicted): TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"\nAccuracy                       : {acc:.3f}   (majority-class baseline = {majority_acc:.3f})")
print(f"Precision ('{POSITIVE}')   : {prec:.3f}   (of {n_pos_pred} predicted Not Completed)")
print(f"Recall    ('{POSITIVE}')   : {rec:.3f}   (of {n_pos_true} actual Not Completed)")
print(f"F1-score  ('{POSITIVE}')   : {f1:.3f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

# overfitting check + cross-validation for a more stable estimate
train_acc = accuracy_score(y_train, pipe.predict(X_train))
print(f"Train accuracy = {train_acc:.3f} vs Test accuracy = {acc:.3f}")
if y.value_counts().min() >= 5:
    cv = cross_validate(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                        scoring={"acc": "accuracy",
                                 "f1_nc": make_scorer(f1_score, pos_label=POSITIVE, zero_division=0),
                                 "rec_nc": make_scorer(recall_score, pos_label=POSITIVE, zero_division=0)})
    print(f"5-fold CV: accuracy {cv['test_acc'].mean():.3f} (+/- {cv['test_acc'].std():.3f}) | "
          f"F1 {cv['test_f1_nc'].mean():.3f} | recall {cv['test_rec_nc'].mean():.3f}")

# ### Task 8 — Is the model reliable enough for administrator decision support?
# The code below states the verdict from the **actual metrics** (so the wording always matches the results), followed by the limitation and improvement.

# ---------------- Task 8: reliability assessment ----------------
print("RELIABILITY ASSESSMENT")
print(f"- Test set: {len(y_test)} reservations ({n_pos_true} actual Not Completed).")
if y_test.nunique() < 2:
    print("- The test set has ONE class only: accuracy is shown but precision/recall/F1 for the absent class are undefined (reported as NaN).")
    print("  Handling: use stratified splitting / cross-validation, or collect more data, before judging the model.")
elif np.isnan(prec) or np.isnan(rec):
    print("- Precision or recall is undefined (no predicted or no actual positives) - reported as NaN, not as a false 0 or 1.")
else:
    beats_baseline = acc > majority_acc
    print(f"- Accuracy {acc:.2f} vs majority-class baseline {majority_acc:.2f} -> "
          + ("better than always guessing the majority class." if beats_baseline else "NOT better than always guessing the majority class."))
    print(f"- For Not Completed: precision {prec:.2f}, recall {rec:.2f}, F1 {f1:.2f}.")
    reliable = (prec >= 0.70) and (rec >= 0.70) and beats_baseline
    if reliable:
        print("VERDICT: acceptable as an ADVISORY aid, still with human review.")
    else:
        print("VERDICT: NOT reliable enough for automatic decisions (precision and/or recall below 0.70 target). "
              "Use only as a soft advisory flag (e.g. send a reminder), never to reject a booking.")
print(f"- Missed risky bookings (false negatives): {fn}; false alarms (false positives): {fp}.")

# **Limitation:** the data is a *small, synthetic sample* (≈ 400 transactions; the test set has only ~80 rows), so metrics are noisy and may not hold on real campus data. Important real-world drivers (weather, exam weeks, event type, previous cancellation history of the user) are not captured, and the classes are imbalanced, so accuracy alone can look good while risky bookings are missed.
#
# **Improvement:** collect more months of real data, add behavioural features known at submission (the requester's *past* cancellation/no-show rate, event type, semester week), compare against Random Forest / Gradient Boosting and tune the decision threshold with cross-validation; then re-evaluate precision/recall for the *Not Completed* class and monitor the model for drift after deployment.

# ---
# # PART V — Requirements Analysis and Use Case Modelling (15 pts)
# ### Task 9 — Functional requirements grounded in actual findings
# The table below is **built from the results computed in Parts III and IV**, so every requirement cites a real number from this notebook rather than an assumed pattern.

# ---------------- Task 9: evidence-based functional requirements ----------------
ev1 = (f"Part III Q2: '{top_user}' has the highest not-completed rate, {user_tbl.loc[top_user,'pct_not_completed']:.1f}% "
       f"({int(user_tbl.loc[top_user,'not_completed'])}/{int(user_tbl.loc[top_user,'n'])} reservations with known outcome).")
ev2 = (f"Part III Q3: '{hi_l}' bookings have the highest not-completed rate, {lead_ct.loc[hi_l,'pct_not_completed']:.1f}% "
       f"({int(lead_ct.loc[hi_l,'Not Completed'])}/{int(lead_ct.loc[hi_l,'n'])}), vs {lead_ct.loc[lo_l,'pct_not_completed']:.1f}% for '{lo_l}'.")
rec_txt  = "undefined" if np.isnan(rec)  else f"{rec:.2f}"
prec_txt = "undefined" if np.isnan(prec) else f"{prec:.2f}"
ev3 = (f"Part IV: Decision Tree (test n = {len(y_test)}) - recall {rec_txt}, precision {prec_txt} for Not Completed; "
       f"top predictors: {', '.join(top3)}.")

fr = pd.DataFrame([
    [ev1, "FR-DM-01",
     f"The system shall send an automated reminder to requesters in the '{top_user}' category before their reserved date and shall require them to confirm or cancel the reservation."],
    [ev2, "FR-DM-02",
     f"The system shall prompt requesters of '{hi_l}' reservations to re-confirm their booking before the reserved date and shall notify the facility administrator of reservations that remain unconfirmed."],
    [ev3, "FR-DM-03",
     "The system shall display to the facility administrator a predicted cancellation/no-show risk level (Low, Medium, High) for each pending reservation, labelled as a prediction, together with the main factors behind it, without automatically approving or rejecting the reservation."],
], columns=["Finding / Evidence", "Requirement ID", "Functional Requirement"])
pd.set_option("display.max_colwidth", None)
display(fr)

# ### Task 10 — Updated UML Use Case Diagram
# The UML use case diagram is **drawn by the code cell below** (inline preview + `Use_Case_Diagram.png`) and the same cell also **creates the editable draw.io file `Facility_Reservation_Use_Case_Diagram.drawio`**. To edit it, open https://app.diagrams.net → *File → Open from device*; then *File → Export as → PNG/PDF* for the report.
#
# | Item | Content |
# |---|---|
# | **System boundary** | Campus Facility Reservation and Monitoring System |
# | **Actors** | Requester (Student / Faculty / Staff / Organization), Facility Administrator, Data Analyst |
# | **Existing use cases** | Search Facility Availability, Submit Reservation Request, Validate Reservation Data, Cancel Reservation, Review and Approve Reservation, Record Actual Attendance |
# | **New analytical use cases (green)** | **View Reservation Analytics Dashboard**, **Predict Cancellation Risk**, **View Risk Explanation**, **Retrain and Evaluate Risk Model**, **Clean and Prepare Training Data** |
#
# **Justification of the new analytical functions**
# * *View Reservation Analytics Dashboard* — supports FR-DM-01/02 by showing cancellation patterns observed in Part III.
# * *Predict Cancellation Risk* — implements FR-DM-03; `«extend»` *Review and Approve Reservation* because the prediction is **optional** and only triggered when the administrator reviews a request (the review works without it).
# * *View Risk Explanation* — `«extend»` *Predict Cancellation Risk*: shown only when the administrator asks why a risk level was given.
# * *Retrain and Evaluate Risk Model* — `«include»` *Clean and Prepare Training Data*: retraining can **never** happen without the cleaning steps of Part II, so the inclusion is mandatory.
# * *Submit Reservation Request* `«include»` *Validate Reservation Data* — validation always runs on every submission (Task 3 validation check).

# ---------------- Task 10: UML Use Case Diagram (inline preview + draw.io file) ----------------
from xml.sax.saxutils import escape
import textwrap
from matplotlib.patches import Ellipse, Rectangle

SYSTEM_NAME = "Campus Facility Reservation and Monitoring System"
DRAWIO_FILE, PNG_FILE = "Facility_Reservation_Use_Case_Diagram.drawio", "Use_Case_Diagram.png"

# kind: actor | old (existing use case) | new (analytical use case)      geometry = x, y, width, height
nodes = {
    "req":   ("Requester\n(Student / Faculty /\nStaff / Organization)", "actor",  70, 330,  40,  80),
    "adm":   ("Facility\nAdministrator",                               "actor", 860, 330,  40,  80),
    "ana":   ("Data Analyst\n(new analytical role)",                   "actor", 860, 690,  40,  80),
    "search":("Search Facility Availability",           "old", 225, 100, 210, 64),
    "submit":("Submit Reservation Request",             "old", 225, 215, 210, 64),
    "valid": ("Validate Reservation Data",              "old", 225, 330, 210, 64),
    "cancel":("Cancel Reservation",                     "old", 225, 445, 210, 64),
    "att":   ("Record Actual Attendance",               "old", 530, 100, 210, 64),
    "review":("Review and Approve Reservation",         "old", 530, 215, 210, 64),
    "pred":  ("Predict Cancellation Risk",              "new", 530, 330, 210, 64),
    "expl":  ("View Risk Explanation",                  "new", 530, 445, 210, 64),
    "dash":  ("View Reservation Analytics Dashboard",   "new", 530, 560, 210, 64),
    "retr":  ("Retrain and Evaluate Risk Model",        "new", 530, 675, 210, 64),
    "clean": ("Clean and Prepare Training Data",        "new", 225, 675, 210, 64),
}
edges = [  # (from, to, type)   include: base -> included | extend: extending -> base
    ("req","search","assoc"), ("req","submit","assoc"), ("req","cancel","assoc"),
    ("adm","att","assoc"), ("adm","review","assoc"), ("adm","pred","assoc"), ("adm","dash","assoc"),
    ("ana","dash","assoc"), ("ana","retr","assoc"),
    ("submit","valid","include"), ("pred","review","extend"), ("expl","pred","extend"), ("retr","clean","include"),
]
BOUNDARY = (200, 50, 580, 760)

# ---------- 1) write the editable draw.io file ----------
def build_drawio():
    cells, ids = [], {k: f"n{i+2}" for i, k in enumerate(nodes)}
    ACT = "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;fontStyle=1;fontSize=12;"
    OLD = "ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;"
    NEW = "ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=2;fontSize=12;fontStyle=1;"
    bx, by, bw, bh = BOUNDARY
    cells.append('<mxCell id="title" value="%s" style="text;html=1;align=center;verticalAlign=middle;fontSize=16;" vertex="1" parent="1"><mxGeometry x="120" y="10" width="760" height="30" as="geometry"/></mxCell>'
                 % escape("<b>Updated Use Case Diagram - " + SYSTEM_NAME + "</b>"))
    cells.append('<mxCell id="boundary" value="%s" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;strokeWidth=2;verticalAlign=top;align=center;fontSize=13;" vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                 % (escape("<b>" + SYSTEM_NAME + "</b>"), bx, by, bw, bh))
    for k, (label, kind, x, y, w, h) in nodes.items():
        style = ACT if kind == "actor" else (NEW if kind == "new" else OLD)
        cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                     % (ids[k], escape(label.replace("\n", "<br>")), style, x, y, w, h))
    for j, (a, b, t) in enumerate(edges):
        if t == "assoc":
            cells.append('<mxCell id="e%d" style="endArrow=none;html=1;rounded=0;strokeColor=#333333;" edge="1" parent="1" source="%s" target="%s"><mxGeometry relative="1" as="geometry"/></mxCell>' % (j, ids[a], ids[b]))
        else:
            cells.append('<mxCell id="e%d" value="\u00ab%s\u00bb" style="endArrow=open;endFill=0;dashed=1;html=1;rounded=0;strokeColor=#333333;fontSize=11;fontStyle=2;labelBackgroundColor=#ffffff;" edge="1" parent="1" source="%s" target="%s"><mxGeometry relative="1" as="geometry"/></mxCell>' % (j, t, ids[a], ids[b]))
    cells.append('<mxCell id="lg1" value="Existing use case" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;" vertex="1" parent="1"><mxGeometry x="200" y="835" width="130" height="40" as="geometry"/></mxCell>')
    cells.append('<mxCell id="lg2" value="NEW analytical use case" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=2;fontSize=10;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="350" y="835" width="150" height="40" as="geometry"/></mxCell>')
    cells.append('<mxCell id="lg3" value="%s" style="text;html=1;align=left;verticalAlign=middle;fontSize=10;" vertex="1" parent="1"><mxGeometry x="520" y="825" width="330" height="60" as="geometry"/></mxCell>'
                 % escape("Solid line = association<br>Dashed arrow = \u00abinclude\u00bb / \u00abextend\u00bb<br>(include: base \u2192 included; extend: extending \u2192 base)"))
    return ('<mxfile host="app.diagrams.net"><diagram id="uc1" name="Use Case Diagram"><mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" '
            'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="920" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')

with open(DRAWIO_FILE, "w", encoding="utf-8") as f:
    f.write(build_drawio())

# ---------- 2) draw the same UML inline (and save as PNG) ----------
fig, ax = plt.subplots(figsize=(11, 9.5))
ax.set_xlim(30, 1010); ax.set_ylim(900, 20); ax.axis("off")
bx, by, bw, bh = BOUNDARY
ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, lw=2))
ax.text(bx + bw/2, by + 20, SYSTEM_NAME, ha="center", fontweight="bold", fontsize=10)
ax.set_title("Updated Use Case Diagram - " + SYSTEM_NAME, fontsize=12, fontweight="bold")
ctr = {}
for k, (label, kind, x, y, w, h) in nodes.items():
    cx, cy = x + w/2, y + h/2; ctr[k] = (cx, cy)
    if kind == "actor":
        ax.plot(cx, y+10, "ko", ms=9, mfc="white", zorder=3)
        ax.plot([cx, cx], [y+15, y+50], "k", zorder=3); ax.plot([cx-18, cx+18], [y+28, y+28], "k", zorder=3)
        ax.plot([cx, cx-15], [y+50, y+75], "k", zorder=3); ax.plot([cx, cx+15], [y+50, y+75], "k", zorder=3)
        ax.text(cx, y+h+6, label, ha="center", va="top", fontsize=8, fontweight="bold")
    else:
        new = kind == "new"
        ax.add_patch(Ellipse((cx, cy), w, h, fc="#d5e8d4" if new else "#dae8fc", ec="#82b366" if new else "#6c8ebf",
                             lw=2 if new else 1, zorder=2))
        ax.text(cx, cy, textwrap.fill(label, 24), ha="center", va="center", fontsize=8,
                fontweight="bold" if new else "normal", zorder=3)
for a, b, t in edges:
    (x1, y1), (x2, y2) = ctr[a], ctr[b]
    if t == "assoc":
        ax.plot([x1, x2], [y1, y2], color="#333333", lw=1, zorder=1)
    else:
        def edge_pt(c, size, toward):          # point where the line c->toward leaves the ellipse
            dx, dy = toward[0] - c[0], toward[1] - c[1]
            k = 1 / np.sqrt((dx / (size[0] / 2)) ** 2 + (dy / (size[1] / 2)) ** 2)
            return c[0] + k * dx, c[1] + k * dy
        pa = edge_pt((x1, y1), nodes[a][4:6], (x2, y2)); pb = edge_pt((x2, y2), nodes[b][4:6], (x1, y1))
        ax.annotate("", xy=pb, xytext=pa, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", ls="--", color="#333333", fc="white", shrinkA=0, shrinkB=0, mutation_scale=14))
        ax.text((x1+x2)/2, (y1+y2)/2, "\u00ab" + t + "\u00bb", ha="center", va="center", fontsize=8, style="italic",
                bbox=dict(fc="white", ec="none", pad=1), zorder=4)
ax.legend(handles=[Patch(fc="#dae8fc", ec="#6c8ebf", label="Existing use case"),
                   Patch(fc="#d5e8d4", ec="#82b366", label="NEW analytical use case")],
          loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(); fig.savefig(PNG_FILE, dpi=150, bbox_inches="tight"); plt.show()
print(f"Created: {DRAWIO_FILE} (editable in draw.io) and {PNG_FILE}")

# ---
# # PART VI — Decision Support and Critical Analysis (10 pts)
# ### Task 11 — Facility administrator dashboard
# The wireframe below is drawn from the notebook's real results. **Solid blue/red panels = OBSERVED facts** (historical, cleaned data). **Orange hatched panels = PREDICTED** values from the model (test-set reservations, i.e. data the model had not seen).

# ---------------- risk scores for the unseen test reservations ----------------
pos_idx = list(pipe.classes_).index(POSITIVE)
proba = pipe.predict_proba(X_test)[:, pos_idx]
risk = pd.cut(proba, bins=[-0.01, 0.35, 0.60, 1.01], labels=["Low", "Medium", "High"])
scored = model_df.loc[X_test.index, ["Reservation_ID", "Facility_Type", "User_Category", "Lead_Time_Days", "Participants"]].copy()
scored["Risk_Probability"] = proba.round(2)
scored["Risk_Level"] = risk.astype(str)
scored = scored.sort_values("Risk_Probability", ascending=False)

def explain_prediction(row_df):
    """Return the human-readable decision-path rules that led to this prediction."""
    Xt = pipe.named_steps["prep"].transform(row_df)
    node_ids = tree.decision_path(Xt).indices
    rules = []
    for nid in node_ids:
        if tree.tree_.children_left[nid] == tree.tree_.children_right[nid]:
            continue                                        # leaf
        f, thr = tree.tree_.feature[nid], tree.tree_.threshold[nid]
        val = Xt[0, f]; nm = clean_names[f]
        if nm in NUM_FEATURES:
            rules.append(f"{nm} {'<=' if val <= thr else '>'} {thr:.1f}")
        else:
            orig = original_feature(nm); level = nm[len(orig) + 1:]
            rules.append(f"{orig} {'is' if val > thr else 'is not'} {level}")
    return rules

print("Risk-level distribution on the test set (PREDICTED):")
print(scored["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int).to_string())
top_row = scored.iloc[0]
print(f"\nExample explanation for {top_row['Reservation_ID']} (risk {top_row['Risk_Probability']:.0%}):")
for r in explain_prediction(X_test.loc[[top_row.name]]): print("  -", r)

# ---------------- Dashboard wireframe (matplotlib) ----------------
OBS, PRED = "#4C78A8", "#F28E2B"
fig = plt.figure(figsize=(17, 10.5)); fig.patch.set_facecolor("white")
gs = GridSpec(4, 6, figure=fig, height_ratios=[0.7, 1.7, 1.7, 1.3], hspace=0.85, wspace=0.7)
fig.subplots_adjust(top=0.92, bottom=0.03, left=0.05, right=0.97)
fig.suptitle("FACILITY ADMINISTRATOR DASHBOARD - Campus Facility Reservation and Monitoring System",
             fontsize=15, fontweight="bold", y=0.985)

# --- KPI tiles (OBSERVED) ---
kpis = [("Total reservations\n(cleaned, unique)", f"{len(df)}", OBS),
        ("Not-completed rate\n(cancelled + no-show)", f"{100*(eda['Reservation_Outcome']=='Not Completed').mean():.1f}%\nof {N_OUT}", "#E45756"),
        ("Avg attendance ratio\n(Used bookings only)", f"{df.loc[df['Usage_Status']=='Used','Attendance_Ratio'].mean():.2f}\nof {int(df.loc[df['Usage_Status']=='Used','Attendance_Ratio'].notna().sum())}", OBS),
        ("Records with unknown\noutcome (excluded)", f"{int(df['Reservation_Outcome'].isna().sum())}", "#999999")]
for i, (lab, val, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i if i < 3 else 3]) if i < 3 else fig.add_subplot(gs[0, 3:4])
    ax.axis("off"); ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=col, lw=2, transform=ax.transAxes))
    ax.text(0.5, 0.64, val, ha="center", va="center", fontsize=11, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.12, lab, ha="center", va="center", fontsize=7, transform=ax.transAxes)
ax = fig.add_subplot(gs[0, 4:6]); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=PRED, lw=2, ls="--", transform=ax.transAxes))
ax.text(0.5, 0.64, f"{int((scored['Risk_Level']=='High').sum())} High-risk", ha="center", fontsize=11, fontweight="bold", color=PRED, transform=ax.transAxes)
ax.text(0.5, 0.15, f"PREDICTED, of {len(scored)} unseen test reservations", ha="center", fontsize=7.5, transform=ax.transAxes)

# --- Observed charts ---
ax = fig.add_subplot(gs[1, 0:2]); ax.bar(fac_counts.index, fac_counts.values, color=OBS)
ax.set_title("OBSERVED: Reservations by facility type\n(n = %d)" % N_FAC, fontsize=9); ax.tick_params(axis="x", rotation=35, labelsize=7); ax.set_ylabel("Reservations", fontsize=8)
ax = fig.add_subplot(gs[1, 2:4]); ax.bar(user_tbl.index, user_tbl["pct_not_completed"], color="#E45756")
ax.set_title("OBSERVED: % not completed by user category\n(denominator: known outcome per group)", fontsize=9); ax.tick_params(axis="x", rotation=35, labelsize=7); ax.set_ylabel("% not completed", fontsize=8)
ax = fig.add_subplot(gs[1, 4:6]); ax.bar(order, lead_ct["pct_not_completed"], color="#E45756")
ax.set_title("OBSERVED: % not completed by lead-time category", fontsize=9); ax.tick_params(axis="x", rotation=20, labelsize=7); ax.set_ylabel("% not completed", fontsize=8)

# --- Predicted risk distribution ---
ax = fig.add_subplot(gs[2, 0:2])
rc = scored["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
ax.bar(rc.index, rc.values, color=PRED, hatch="//", edgecolor="white")
ax.set_title("PREDICTED: risk-level distribution\n(model output on unseen test reservations)", fontsize=9, color="#B25F00")
ax.set_ylabel("Reservations", fontsize=8)
for k, v in enumerate(rc.values): ax.text(k, v + 0.3, int(v), ha="center", fontsize=8)

# --- High-risk table (predicted) ---
ax = fig.add_subplot(gs[2, 2:6]); ax.axis("off")
ax.set_title("PREDICTED: highest-risk pending reservations (advisory only)", fontsize=9, color="#B25F00", loc="left")
hdr = f"{'ID':<10}{'Facility':<17}{'User':<22}{'Lead(d)':>8}{'Pax':>6}{'Risk':>8}  Level"
lines = [hdr, "-" * 82] + [f"{r.Reservation_ID:<10}{r.Facility_Type:<17}{r.User_Category:<22}{r.Lead_Time_Days:>8.0f}"
                          f"{(0 if pd.isna(r.Participants) else r.Participants):>6.0f}{r.Risk_Probability:>8.0%}  {r.Risk_Level}"
                          for r in scored.head(6).itertuples()]
ax.text(0, 0.95, "\n".join(lines), family="monospace", fontsize=8, va="top", transform=ax.transAxes)

# --- Explanation panel + legend ---
ax = fig.add_subplot(gs[3, 0:4]); ax.axis("off")
why = explain_prediction(X_test.loc[[top_row.name]])
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=PRED, lw=1.5, ls="--", transform=ax.transAxes))
ax.text(0.02, 0.93, f"WHY is {top_row['Reservation_ID']} high risk? (PREDICTED - risk {top_row['Risk_Probability']:.0%})",
        fontsize=9, fontweight="bold", color="#B25F00", va="top", transform=ax.transAxes)
ax.text(0.02, 0.75, "Decision-path factors:\n" + "\n".join("  - " + r for r in why[:5]) +
        "\n\nAdministrator actions:  [Send reminder]  [Request confirmation]  [Approve anyway]  [Review manually]",
        fontsize=8, va="top", transform=ax.transAxes)
ax = fig.add_subplot(gs[3, 4:6]); ax.axis("off")
ax.legend(handles=[Patch(fc=OBS, label="OBSERVED fact (historical data)"), Patch(fc="#E45756", label="OBSERVED cancellation measure"),
                   Patch(fc=PRED, hatch="//", ec="white", label="PREDICTED (model output)")],
          loc="upper left", fontsize=8, title="Legend", frameon=True)
ax.text(0, 0.0, "Predictions are probabilities, not facts.\nThey support - never replace - the\nadministrator's decision.", fontsize=8, va="bottom", transform=ax.transAxes)
plt.show()

# ### Task 12 — Short written answers
#
# **(a) Why should predicted high risk not automatically reject a reservation?**
# A prediction is a *probability based on past patterns*, not a fact about this particular booking. The model makes errors (false positives would unfairly deny people who would have used the facility), it only sees a few variables, and group-level patterns (e.g. a user category) must not become discrimination against individuals. Automatic rejection would also create unused capacity and unfairness with no appeal. A high-risk flag should instead trigger *softer actions* — a reminder, a confirmation request, or administrator review — and the human administrator makes the final decision.
#
# **(b) How does poor data quality affect the model and the information system?**
# In this exam's data, duplicates inflate counts, inconsistent labels (`Canceled` vs `Cancelled`) split one category into several, invalid values (999 participants, negative lead time) distort statistics, and missing outcomes make labels unknowable. The model then learns wrong patterns ("garbage in, garbage out"), accuracy figures become misleading, and administrators lose trust in dashboards and make poor staffing/allocation decisions. For the information system itself, bad data shows that **input validation and controlled vocabularies (drop-down lists)** are missing at data entry, so quality must be enforced at the source, not only cleaned later.
#
# **(c) What information should be displayed to explain a high-risk prediction?**
# (1) The **risk level and probability**, clearly labelled *Predicted*; (2) the **top contributing factors** in plain words (e.g. user category, lead time, facility type, weekend booking — see the decision-path example above); (3) **historical evidence** for those factors with denominators (e.g. "x of n similar bookings were not completed"); (4) the **model's reliability** (test accuracy/recall and date of last training); (5) any **missing or uncertain inputs**; and (6) **suggested actions** with the final approve/reject choice left to the administrator.

# ---
# ## Final Step — Save and download the deliverables (CSV files + UML diagram)

# ---------------- Save + download deliverables ----------------
print("Files created in", os.getcwd())
DELIVERABLES = [RAW_CSV, CLEAN_CSV, DRAWIO_FILE, PNG_FILE]
for f in DELIVERABLES:
    print(f"  {f:45s} {os.path.getsize(f):>8,d} bytes")

try:                                    # works inside Google Colab only
    from google.colab import files
    for f in DELIVERABLES: files.download(f)
except Exception:
    print("\n(Not running in Colab - use the file browser on the left to download the files.)")

print("\nSubmission checklist: this .ipynb | facility_reservations_clean.csv | the .drawio use case diagram + Use_Case_Diagram.png | PDF report")
