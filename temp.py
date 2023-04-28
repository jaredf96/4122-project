# Import necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Loading the Datasets
df_ford = pd.read_csv('ford.csv')
df_Volkswagen = pd.read_csv('vw.csv')

# Cleaning the datset
def data_cleaning(data):
    numeric_columns = ['price', 'year', 'mileage', 'tax', 'mpg', 'engineSize']
    numeric_data = data[numeric_columns]
    # Dropping null values from the numeric columns
    numeric_data.dropna(inplace=True)
    cleaned_data = pd.concat([numeric_data, data.drop(columns=numeric_columns)], axis=1)
    return cleaned_data


# Clean the datasets
df_ford_cleaned = data_cleaning(df_ford)
df_vw_cleaned = data_cleaning(df_Volkswagen)

# Combine both datasets into a single dataframe
data = pd.concat([df_ford_cleaned, df_vw_cleaned])

st.title("Ford vs Volkswagen Used Car Data Analysis")
menuBar = option_menu(None, ["Visualizations", "Predictions"], 
    icons=['bar-chart', 'bullseye'], 
    menu_icon="cast", orientation="horizontal")


# User input for selecting dataset
st.sidebar.header("Select Dataset(s)")
datasets_merged = {'Ford': df_ford_cleaned, 'Volkswagen': df_vw_cleaned}
datasets_selected = st.sidebar.multiselect("Choose the dataset(s) you're interested in:", list(datasets_merged.keys()), default=['Ford'])


# Create visualizations to answer the questions
def create_visualizations(data, manufacturer, col):
    col.header(f"{manufacturer} Car Analysis")
    
    # Apply the price filter
    filtered_data = data[(data['price'] >= price_range[0]) & (data['price'] <= price_range[1])]

    # Number of cars per model
    col.subheader("Number of Cars per Model")
    fig, ax = plt.subplots()
    car_model_count = filtered_data['model'].value_counts()
    sns.barplot(x=car_model_count.index, y=car_model_count.values,color='red', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Cars")
    ax.set_title("Number of Cars per Model")
    col.pyplot(fig)

    # Average price by year
    col.subheader("Average Price Variation By Year")
    fig, ax = plt.subplots()
    sns.lineplot(data=filtered_data, x='year', y='price', ci=None, color='red', ax=ax)
    ax.set_title("Average Price Variation By Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    col.pyplot(fig)
    
    # Average mpg by fuel type
    col.subheader("Average MPG by Fuel Type")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_data, x='fuelType', y='mpg', ci=None, color='red', ax=ax)
    ax.set_title("Average MPG by Fuel Type")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Mile per Gallon")
    col.pyplot(fig)

    # Distribution of transmission types
    col.subheader("Distribution of Transmission Types")
    fig, ax = plt.subplots()
    filtered_data['transmission'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    ax.set_title("Distribution of Transmission Types")
    col.pyplot(fig)
    
    # Best car model for the user's price range (horizontal bar plot)
    col.subheader("Best Car Models in the Selected Price Range")
    filtered_data_by_price = filtered_data.groupby('model')[['price']].mean().sort_values('price')
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_data_by_price, x='price', y=filtered_data_by_price.index, color='red', ax=ax)
    ax.set_title(f"Best Car Models in the Selected Price Range ({manufacturer} cars only)")
    ax.set_xlabel("Price")
    ax.set_ylabel("Model")
    col.pyplot(fig)
# Display visualizations and predictions based on the selected tab
if menuBar == "Visualizations":
    # Price filter slider
    min_price = int(min(df_ford['price'].min(), df_Volkswagen['price'].min()))
    max_price = int(max(df_ford['price'].max(), df_Volkswagen['price'].max()))
    price_range = st.sidebar.slider("Select the Price Range", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=100)
    if len(datasets_selected) == 2:
        left_column, right_column = st.columns(2)
        create_visualizations(datasets_merged[datasets_selected[0]], datasets_selected[0], left_column)
        create_visualizations(datasets_merged[datasets_selected[1]], datasets_selected[1], right_column)
    else:
        for dataset in datasets_selected:
            create_visualizations(datasets_merged[dataset], dataset, st)

            
# Inputs for pridicting the Car Price
def display_sidebar_inputs():
    model = st.selectbox("Model", data['model'].unique())
    transmission = st.radio("Transmission", data['transmission'].unique())
    fuelType = st.radio("Fuel Type", data['fuelType'].unique())
    year = st.slider("Year", int(data['year'].min()), int(data['year'].max()), int(data['year'].mean()))
    mileage = st.number_input("Mileage", int(data['mileage'].min()), int(data['mileage'].max()), int(data['mileage'].mean()),step=100)
    tax = st.slider("Tax", int(data['tax'].min()), int(data['tax'].max()), int(data['tax'].mean()))
    mpg = st.slider("Miles per Gallon (MPG)", int(data['mpg'].min()), int(data['mpg'].max()), int(data['mpg'].mean()),step=10)
    engineSize = st.slider("Engine Size", float(data['engineSize'].min()), float(data['engineSize'].max()), float(data['engineSize'].mean()), step=0.5)
    return model, transmission, fuelType, year, mileage, tax, mpg, engineSize

if menuBar == "Predictions":
    # Display the sidebar user inputs
    model, transmission, fuelType, year, mileage, tax, mpg, engineSize = display_sidebar_inputs()
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
    "model": [model],
    "transmission": [transmission],
    "fuelType": [fuelType],
    "year": [year],
    "mileage": [mileage],
    "tax": [tax],
    "mpg": [mpg],
    "engineSize": [engineSize],
    })
    # Loading the model
    with open('ford_and_Volkswagen_price_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    # Predict the price using the model
    predicted_price = model.predict(input_data)
    # Display the predicted price
    st.write(f"The predicted price for the selected Car Data is: ${predicted_price[0]}")
    
if menuBar == "Dataset Info":
    if len(datasets_selected) == 2:
        column_1, column_2 = st.columns(2)
        display_dataset_info(df_ford, "Ford", column_1)
        display_dataset_info(df_Volkswagen, "Volkswagen", column_2)
    else:
        if "Ford" in datasets_selected:
            display_dataset_info(df_ford, "Ford", st)
        if "Volkswagen" in datasets_selected:
            display_dataset_info(df_Volkswagen, "Volkswagen", st)

   
