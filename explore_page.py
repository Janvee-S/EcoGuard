import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and preprocess the dataset
@st.cache_data
def load_data():
    # Assuming the dataset is saved as "GreenZoneData.csv"
    df = pd.read_csv("GreenZoneData.csv")

    # Keep only relevant columns and drop missing values
    df = df[["so2", "no2", "rspm", "Air Quality Index", "Green Zones"]]
    df = df.dropna()
    df["Green Zones"] = df["Green Zones"].astype(int)  # Ensure binary outcome is int (0 or 1)
    
    return df

# Load the data
df = load_data()

# Streamlit page to explore air quality data
def show_explore_page():
    st.title("Explore Green Zone Data")

    st.write(
        """
    ### Air Quality and Green Zone Classification
    Explore how different air quality metrics relate to whether a location is classified as a Green Zone.
    """
    )

    # Pie chart to show distribution of Green Zones and Non-Green Zones
    data = df["Green Zones"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=["Green Zone", "Not Green Zone"], autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Percentage of Green Zones vs Non-Green Zones""")
    st.pyplot(fig1)

    # Bar chart to show average SO2, NO2, RSPM based on whether it's a Green Zone
    st.write(
        """
    #### Mean Air Quality Metrics Based on Green Zone Classification
    """
    )

    air_quality_metrics = df.groupby("Green Zones")[["so2", "no2", "rspm", "Air Quality Index"]].mean()

    st.bar_chart(air_quality_metrics)

    # Line chart to show how air quality index varies across the dataset
    st.write(
        """
    #### Air Quality Index Over Different Samples
    """
    )

    st.line_chart(df["Air Quality Index"])

# Call the function to show the exploration page
show_explore_page()
