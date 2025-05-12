import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import scipy.fftpack as fft
import mplcursors
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
from datetime import datetime 
import json

# --- Configuration ---
ZABBIX_URL = "https://notify.airtel.com/zabbix/api_jsonrpc.php"
AUTH_TOKEN = "b8ed64b0edc2d7e754024faed5d1d6000b2ebd6b934186d7b82017c8db4a2f37"
ITEM_IDS = [45380239, 45380257]  # ECOM, UCM, and ESB


ITEM_MAPPING = {
    "45380239": "ecom",
    "45380257": "ucm",
}


# --- Fetch Data from Zabbix API ---
def fetch_zabbix_data():
    current_time = int(datetime.now(timezone.utc).timestamp())     # Fetches current time in UTC
    time_from = current_time - (2 * 24 * 60 * 60)  # Last 2 days   
  
    response = requests.get(
        ZABBIX_URL,

        json={
            "jsonrpc": "2.0",
            "method": "history.get",
            "params": {
                "output": "extend",
#                 "history": 0,
                "itemids": ITEM_IDS,
                "time_from": time_from,
                "time_till": current_time,
                "sortfield": "clock",
                "sortorder": "DESC",
#                 "limit": 5000,
            },
            "auth": AUTH_TOKEN,
            "id": 2
        }
    )

    data = response.json()
#     print('------ response.json() data ---------')
#     print(response.json())

    print()
#     print(f"Current Time (Epoch): {current_time}")
    
    if "result" not in data:
        raise ValueError("No data received from Zabbix API!")

    records = []
    for item in data["result"]:
        item_name = ITEM_MAPPING.get(item["itemid"], "unknown")
        timestamp = datetime.utcfromtimestamp(int(item["clock"]))
        records.append({"authority": item_name, "value": round(float(item["value"]), 2), "timestamp": timestamp})



    df = pd.DataFrame(records)
    print()
    print('----------------------------------------------- DataFrame -----------------------------------------------------------')
    print(df)
#     df.set_index("timestamp", inplace=True)
    return df

# --- Detect Seasonality Using FFT ---
def detect_seasonality_fft(df):
    
    values = df["value"].values
    n = len(values)

    if n < 10:
        print("⚠ Not enough data for FFT analysis. Using default seasonality = 6")
        return 6


    freqs = fft.fft(values)
    power = np.abs(freqs[:n // 2])
    peak_freq = np.argmax(power[1:]) + 1
    seasonality = max(6, min(n // 2, peak_freq))

    print(f"🔍 FFT detected seasonality period: {seasonality}")
    return seasonality


# --- Detect Trend using Linear Regression + Coefficient of Variation---
def detect_trend(df, seasonal_period):        
    # Coefficient of Variation for Trend
    decomposition = seasonal_decompose(df['value'], model='additive', period=seasonal_period)
    trend_component = decomposition.trend.dropna()
    if not trend_component.isnull().all():
        cv_trend = trend_component.std() / trend_component.mean()

        if cv_trend > 0.5:
            return 'mul'
        elif cv_trend > 0.1:
            return 'add'
            
    return None


# --- Detect Seasonality Type ---
def detect_seasonality_type(df, seasonal_period):
    decomposition = seasonal_decompose(df['value'], model='additive', period=seasonal_period)
    seasonal_component = decomposition.seasonal.dropna()

    if seasonal_component.isnull().all():
        return None  

    cv_season = seasonal_component.std() / seasonal_component.mean()
    return 'mul' if cv_season > 0.5 else 'add'


# --- Apply Holt-Winters Model & Detect Anomalies ---
def detect_anomalies(df):
    if df.empty:
        raise ValueError("DataFrame is empty. No data to process!")

    seasonal_period = detect_seasonality_fft(df)
    trend_type = detect_trend(df, seasonal_period) or 'add'
    seasonality_type = detect_seasonality_type(df, seasonal_period) or 'add'

    print(f"📌 Model: trend='{trend_type}', seasonal='{seasonality_type}'")

    # Apply Holt-Winters Model
    model = ExponentialSmoothing(df['value'], trend=trend_type, seasonal=seasonality_type,
                                 seasonal_periods=seasonal_period, initialization_method='estimated')

    hw_fit = model.fit()
    forecast_steps = 10  # Number of future steps to forecast
    future_forecast = hw_fit.forecast(steps=forecast_steps)
#     print("\nFuture Forecasted Values:")
#     print(future_forecast)
    
    df['forecast'] = hw_fit.fittedvalues.round(2)
    df['deviation_percentage'] = abs(df['value'] - df['forecast']) / df['forecast'] * 100
    df['deviation_percentage'] = df['deviation_percentage'].round(2)
    df['is_anomaly'] = df['deviation_percentage'] > 20
    anomalies = df[df['is_anomaly']]

    return df, anomalies, hw_fit


# --- Plot Data ---
def plot_results(df, anomalies):
    plt.figure(figsize=(25, 5))
    plt.plot(df.index, df['value'], label="Observed", marker='o', color='blue')
 # plt.plot(df.index, df['forecast'], label="Fitted (Training)", linestyle="--", color="orange", zorder=3)
    
     # Confidence band range (±20%)
    upper_bound = df['value'] * 1.2
    lower_bound = df['value'] * 0.8

    # Highlight confidence band
    plt.fill_between(df.index, lower_bound, upper_bound, color='green', alpha=0.3, label="Confidence Band (±20%)")
    
 # plt.scatter(anomalies.index, anomalies['value'], color="red", label="Anomaly", zorder=3)
    plt.title("Anomaly Detection on Airtel Data (Traffic)")
    plt.xlabel("Time")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    mplcursors.cursor(hover=True)
    plt.show()
    
    
# --- Run Process ---
df = fetch_zabbix_data()
exit()
df, anomalies, hw_fit = detect_anomalies(df)
plot_results(df, anomalies)


# --- Print Updated Data with Forecast ---
print("\n✅ **Final Data with Forecasted Values:**")
print(df[['authority', 'value', 'forecast']])
filtered_df = df.reset_index()[['timestamp', 'authority', 'value', 'forecast']]
json_output = filtered_df.to_json(orient='records', date_format='iso')
print()
print('-------------------- JSON Output ---------------------')
print(json_output)
print()
print(f"\n✅ **Anomaly Detection Completed! {len(anomalies)} anomalies found.**")
print("\n----- **Anomalies Detected** -----")
print(anomalies[['authority', 'value', 'forecast', 'deviation_percentage']])
print()


# --- Dump json data to ES index ---
def write_id_elasticsearch(timestamp, authority, value, forecast):
    print('Elastic Connection Started')
    headers = {'Content-Type': 'application/json'}
    doc = {
           "@timestamp": timestamp,
           'authority': authority,
           'value': value,
           'forecast': forecast,
          }


    # Making a request to Elasticsearch
    response = requests.post(
        'http://central-elastic.airtel.com:9200/monitoring_forecast-logs-25/_doc',
        headers=headers,
        json=doc,
        auth=('Monitoring','Monitoring@123')
    )


    # Check the response status and handle accordingly
    if response.status_code == 201:
        print("Document indexed successfully.")
    else:
        print(f"Failed to index document: {response.status_code} - {response.text}")


# Convert JSON string to Python list
data = json.loads(json_output)

# Looping through the data list
for item in data:
    try:
        write_id_elasticsearch(
            timestamp=item['timestamp'],
            authority=item['authority'],
            value=item['value'],
            forecast=item['forecast']

        )

    except Exception as e:
        print(f"Error indexing document {item}: {e}") 