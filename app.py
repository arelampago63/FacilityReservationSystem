
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Facility Reservation Monitoring System",
    page_icon="🏫",
    layout="wide"
)

st.title("Campus Facility Reservation Monitoring System")
st.write(
    "A student project for monitoring campus facility reservations "
    "and analyzing reservation outcomes."
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Dataset",
        "Data Analysis",
        "Prediction",
        "Dashboard",
        "System Design"
    ]
)

if page == "Home":
    st.header("Welcome")
    st.write("Welcome to the Campus Facility Reservation Monitoring System.")
    st.info("Choose a module from the sidebar to explore the system.")

elif page == "Dataset":
    st.header("Dataset")
    file_path = Path("facility_reservations.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
        st.write("Dataset shape:", df.shape)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Upload facility_reservations.csv to the repository.")

elif page == "Data Analysis":
    st.header("Data Analysis")
    file_path = Path("facility_reservations.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
        st.subheader("Dataset Overview")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
    else:
        st.warning("Dataset not found. Upload facility_reservations.csv.")

elif page == "Prediction":
    st.header("Reservation Outcome Prediction")
    st.write(
        "This module will be connected to the trained model "
        "after the model and dataset files are uploaded."
    )
    st.info("Predictions are not enabled yet.")

elif page == "Dashboard":
    st.header("Dashboard")
    file_path = Path("facility_reservations.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
        col1, col2 = st.columns(2)
        col1.metric("Total Reservations", len(df))

        if "Reservation_Outcome" in df.columns:
            col2.metric(
                "Completed Reservations",
                int((df["Reservation_Outcome"] == "Completed").sum())
            )
            st.subheader("Reservation Outcomes")
            st.bar_chart(df["Reservation_Outcome"].value_counts())
        else:
            col2.metric("Total Columns", len(df.columns))
    else:
        st.warning("Upload facility_reservations.csv to show dashboard data.")

elif page == "System Design":
    st.header("System Design")
    image_path = Path("Use_Case_Diagram.png")

    if image_path.exists():
        st.image(str(image_path), caption="Use Case Diagram")
    else:
        st.info("The use-case diagram has not been uploaded yet.")
