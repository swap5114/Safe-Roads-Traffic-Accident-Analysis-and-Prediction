import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tempfile

data = pd.read_csv("Dataset.csv")
download_data = data
data['MONAT'] = data['MONAT'].astype(str).str[:6]
data['DATUM'] = pd.to_datetime(data['MONAT'], format='%Y%m')


def filter_data_by_year(data, start_year, end_year):
    return data[(data['JAHR'] >= start_year) & (data['JAHR'] <= end_year)]


def preprocessig_data(data, category, type, year, month, scaler):
    
    data['DATUM'] = pd.to_datetime(data['MONAT'], format='%Y%m')

    cutoff_date = pd.to_datetime(f"{year}{month:02d}", format='%Y%m')
    filtered_data = data[
        (data['MONATSZAHL'] == category) &
        (data['AUSPRAEGUNG'] == type) &
        (data['DATUM'] < cutoff_date)
        ]
    if filtered_data.empty:
        return None, None, None, None

    series = filtered_data['WERT'].dropna().values.reshape(-1, 1)
    series_reshaped = series.reshape(-1, 1)
    scaled_data = scaler.fit_transform(series_reshaped)

    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length + 1])
        return np.array(sequences)

    sequence_length = 5  # Number of time steps to look back
    sequences = create_sequences(scaled_data, sequence_length)
    X, y = sequences[:, :-1], sequences[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def plot_historical_data(data):
    with st.spinner("Generating Graph..."):
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='DATUM', y='WERT', hue='MONATSZAHL', data=data, marker='o')
        plt.title('Historical Number of Accidents Per Category')
        plt.xlabel('Date')
        plt.ylabel('Number of Accidents')
        plt.xticks(rotation=45)
        plt.legend(title='Category', loc='upper left')
        plt.tight_layout()

    st.pyplot(plt)

def plot_seasonal_analysis(data):
    monthly_data = data.groupby(['MONATSZAHL', data['DATUM'].dt.month])['WERT'].mean().reset_index()
    plt.figure(figsize=(15, 6))
    sns.lineplot(x='DATUM', y='WERT', hue='MONATSZAHL', data=monthly_data, marker='o')
    plt.title('Seasonal Analysis of Accidents Per Category')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Accidents')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Category', loc='upper left')
    plt.tight_layout()
    st.success('Graph Generated!')
    st.pyplot(plt)

def plot_yearly_analysis(data):
    yearly_data = data.groupby(['MONATSZAHL', 'JAHR'])['WERT'].sum().reset_index()
    plt.figure(figsize=(15, 6))
    sns.barplot(x='JAHR', y='WERT', hue='MONATSZAHL', data=yearly_data)
    plt.title('Comparative Analysis of Accidents Per Category by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Accidents')
    plt.legend(title='Category', loc='upper right')
    plt.tight_layout()
    st.pyplot(plt)


def plot_predicted_graph(predicted, y_test_original):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title('Actual vs. Predicted Accidents')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def train_lstm_model(X_train, y_train, num_layers, num_nodes, epoch):
    model = Sequential()
    model.add(LSTM(num_nodes, return_sequences=(num_layers > 1), input_shape=(X_train.shape[1], 1)))
    for _ in range(1, num_layers):
        model.add(LSTM(num_nodes, return_sequences=(_ < num_layers - 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.1,
              callbacks=[early_stopping])
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
        model.save(tmp.name)
        return model, tmp.name


def main():
    st.title("Traffic Accident Analysis and Prediction")

    option = st.sidebar.selectbox('Select an Option', ('Data Visualization', 'Model Training'))

    if option == 'Data Visualization':
        st.header("Data Visualization")

        with st.sidebar.expander("Graph Descriptions"):
            st.markdown("""
                **Historical Number of Accidents**: Shows the total number of accidents over time, highlighting trends and changes in accident frequency.
                \n**Comparative Analysis by Year**: Compares the total number of accidents per year across different categories, indicating yearly trends and variations.
                \n**Seasonal Analysis of Accidents**: Provides insights into how accidents vary by month, identifying seasonal patterns.
            """)
        st.sidebar.download_button(
            label="Download Dataset",
            data=download_data.to_csv(index=False),
            file_name="Dataset.csv",
            mime="text/csv",
        )

        start_year, end_year = st.select_slider(
            "Select a range of years",
            options=list(range(2000, 2022)),
            value=(2000, 2021)
        )

        if start_year == end_year:
            st.warning("Please select a range of at least one year.")

        filtered_data = filter_data_by_year(data, start_year, end_year)

        graph_option = st.selectbox("Choose the type of graph",
                                    ("Historical Number of Accidents",
                                     "Comparative Analysis by Year",
                                     "Seasonal Analysis of Accidents"))

        submit_button = st.button('Generate Graph')

        if submit_button:
            if graph_option == "Historical Number of Accidents":
                plot_historical_data(filtered_data)
            elif graph_option == "Seasonal Analysis of Accidents":
                plot_seasonal_analysis(filtered_data)
            elif graph_option == "Comparative Analysis by Year":
                plot_yearly_analysis(filtered_data)

    elif option == 'Model Training':
        st.sidebar.download_button(
            label="Download Dataset",
            data=download_data.to_csv(index=False),
            file_name="Dataset.csv",
            mime="text/csv",
        )
        st.header("Model Training")

        category = st.selectbox('Select Category', data['MONATSZAHL'].unique())
        typeof = st.selectbox('Select Type', data['AUSPRAEGUNG'].unique())
        unique_years = sorted([year for year in data['JAHR'].unique() if year < 2022], reverse=True)
        year = st.selectbox('Select Year', unique_years)
        # year = st.selectbox('Select Year', sorted([y for y in data['JAHR'].unique() if y not in ["2022", "2023"]], reverse=True))
        month = st.selectbox('Select Month for Prediction', range(1, 13))

        num_layers = st.number_input('Number of LSTM Layers', min_value=1, max_value=10, value=3)
        num_nodes = st.number_input('Number of Nodes per Layer', min_value=10, max_value=500, value=100)
        epochs = st.number_input("Number of Epochs", min_value=10, max_value=500, value=50)

        scaler = MinMaxScaler(feature_range=(0, 1))

        result = preprocessig_data(data, category, typeof, year, month, scaler)

        if result[0] is not None:
            X_train, X_test, y_train, y_test = result
            if st.button('Train Model'):
                print(type(year), type(month))
                with st.spinner('Training in progress...'):
                    model,model_path = train_lstm_model(X_train, y_train, num_layers, num_nodes, epochs)
                    st.success('Training completed. Creating The Comparison Graph!')

                    with open(model_path, "rb") as file:
                        btn = st.download_button(
                            label="Download trained model",
                            data=file,
                            file_name="trained_model.h5",
                            mime="application/octet-stream"
                        )

                    predicted = model.predict(X_test)

                    predicted = scaler.inverse_transform(predicted)
                    y_test_original = scaler.inverse_transform(y_test)
                    plot_predicted_graph(predicted, y_test_original)
        else:
            st.error("Insufficient data for the selected Category and Type")

    st.markdown("---") 
    st.markdown("Made with ❤ by Prashuk!")

if __name__ == "__main__":
    main()
