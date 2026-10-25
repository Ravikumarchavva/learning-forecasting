import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

def main():
    # 1. Load the Nixtla-hosted Air Passengers dataset
    url = 'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv'
    df = pd.read_csv(url)

    # The dataset comes with 'timestamp' and 'value'. Let's format it for Nixtla:
    df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Add a unique_id since we are tracking one series
    df['unique_id'] = 'air_passengers'

    print("First few rows of the formatted dataset:")
    print(df.head())

    # 2. Split into train and test sets (Hold out the last 12 months for validation)
    train_df = df.head(-12)
    test_df = df.tail(12)

    # 3. Instantiate the models
    # Since this is monthly data, the seasonal length is 12
    models = [
        AutoARIMA(season_length=12),
        AutoETS(season_length=12)
    ]

    # 4. Initialize StatsForecast
    # 'freq=MS' stands for Month Start frequency
    sf = StatsForecast(
        models=models,
        freq='MS', 
        n_jobs=-1 # Uses all available CPU cores
    )

    # 5. Fit the models and generate a 12-month forecast
    # 'h' is the forecast horizon
    forecast_df = sf.forecast(df=train_df, h=12)
    print("\nGenerated Forecasts:")
    print(forecast_df.head())

    # 6. Merge with actual values to see how well they performed
    comparison_df = test_df.merge(forecast_df, on=['unique_id', 'ds'])

    # 7. Plot the results
    fig = sf.plot(df, forecast_df, max_insample_length=36, engine='matplotlib')
    fig.savefig('forecast_plot.png', bbox_inches='tight')
    print("\nPlot saved to forecast_plot.png")
    plt.show()

if __name__ == "__main__":
    main()