# 🏫 Campus Facility Reservation and Monitoring System

Streamlit app for the **Systems Analysis and Design (SAD) – Data Mining Laboratory Examination**.
**Author:** April Jhon B. Relampago · BSIS 3A

It walks through the full pipeline: historical data → cleaning → analysis → prediction →
evidence-based requirements → use cases → decision support.

> All data is **synthetic**. No real student information is used.

## Pages
| Page | Content |
|---|---|
| Overview | Summary KPIs and navigation |
| Data & Cleaning | Raw vs clean data, data dictionary, cleaning decisions (Tasks 1–5) |
| Exploratory Analysis | Business questions and charts (Task 6) |
| Model & Reliability | Decision Tree, metrics, confusion matrix, verdict (Tasks 7–8) |
| Risk Predictor | Predict the cancellation/no-show risk of a new reservation |
| Admin Dashboard | Observed facts vs predicted risk (Task 11) |
| Requirements & Use Case | Evidence-based requirements + UML use case diagram (Tasks 9–10) |
| Critical Analysis | Written answers (Task 12) |

## Repository structure
```
FacilityReservationSystem/
├── app.py                          # Streamlit app
├── requirements.txt                # Python dependencies
├── README.md
├── facility_reservations.csv       # raw synthetic dataset (with intentional problems)
├── facility_reservations_clean.csv # cleaned dataset + derived variables
├── trained_model.joblib            # trained Decision Tree pipeline
└── Use_Case_Diagram.png            # updated UML use case diagram
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io> → **New app**.
3. Select the repository and branch, set **Main file path** to `app.py`, then **Deploy**.

## Notes
- The model predicts `Reservation_Outcome` (*Completed* vs *Not Completed*) using 7 predictors known at submission time.
  Post-outcome columns are excluded to avoid data leakage.
- If `trained_model.joblib` cannot be loaded (e.g. a different scikit-learn version), the app automatically re-trains the
  same pipeline from the clean CSV (`random_state = 42`).
- Predictions are probabilities that **support, never replace**, the administrator's decision.
