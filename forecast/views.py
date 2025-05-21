from django.shortcuts import render 
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime, timedelta
import pytz

# API Setup
API_KEY = '6ee2176d038f89416fcc2641262f93fb'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_5_day_forecast(city):
    try:
        url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Ambil 1 data per hari pada jam 12:00
        daily_data = []
        for entry in data['list']:
            if "12:00:00" in entry['dt_txt']:
                daily_data.append({
                    'date': entry['dt_txt'].split()[0],
                    'temp': round(entry['main']['temp']),
                    'description': entry['weather'][0]['description'].capitalize(),
                    'icon': entry['weather'][0]['icon']
                })
                if len(daily_data) == 5:
                    break
        return daily_data

    except Exception as e:
        print(f"Error in 5-day forecast: {e}")
        return None


# Handle current weather API
def get_current_weather(city):
    try:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_speed': data['wind']['speed'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    except (requests.RequestException, KeyError) as e:
        print(f"Error fetching weather: {e}")
        return None

# Load dataset and encode categorical
# Load dataset
full_data = pd.read_csv(r'C:\Users\ASUS\Documents\WeatherCoba\weather.csv')

# Encode WindGustDir (categorical)
wind_encoder = LabelEncoder()
full_data['WindGustDir'] = wind_encoder.fit_transform(full_data['WindGustDir'])

# Encode RainTomorrow (target)
rain_encoder = LabelEncoder()
full_data['RainTomorrow'] = rain_encoder.fit_transform(full_data['RainTomorrow'])

# Features & target
X = full_data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
y = full_data['RainTomorrow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier for RainTomorrow prediction
rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_train, y_train)

# Reusable regression setup
def prepare_regression_data(full_data, feature):
    X, y = [], []
    for i in range(len(full_data) - 1):
        X.append(full_data[feature].iloc[i])
        y.append(full_data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# Main View
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)
        daily_forecasts = get_5_day_forecast(city)



        if not current_weather:
            return render(request, 'weather.html', {'error': 'Gagal mengambil data cuaca.'})

        # Arah angin → arah mata angin
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_encoded = wind_encoder.transform([compass_direction])[0] if compass_direction in wind_encoder.classes_ else -1

        # Prediksi hujan menggunakan model klasifikasi
        current_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_encoded,
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }])
        weather_condition_encoded = rain_model.predict(current_data)[0]
        weather_condition = rain_encoder.inverse_transform([weather_condition_encoded])[0]

        # Prediksi suhu & kelembaban masa depan
        X_temp, y_temp = prepare_regression_data(full_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(full_data, 'Humidity')
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # Waktu prediksi
        current_time = datetime.now()
        future_times = []
        for i in range(5):
            future_time = current_time + timedelta(hours=i+1)
            future_times.append(future_time.strftime("%H:00"))
        # timezone = pytz.timezone('Asia/Karachi')
        # now = datetime.now(timezone)




        # next_hour = now + timedelta(hours=1)
        # next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        # future_times = [(now + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]

        # future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Bungkus ke context
        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': current_time.strftime("%H:%M"),
            'date': current_time.strftime("%B %d, %Y"),
            'wind': current_weather['wind_gust_speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            #'rain_prediction': 'Yes' if rain_prediction == 1 else 'No',
            'weather_condition': weather_condition,
            
        

        }
        
        context['daily_forecasts'] = []
        if daily_forecasts:
            for forecast in daily_forecasts:
                date_obj = datetime.strptime(forecast['date'], "%Y-%m-%d")
                context['daily_forecasts'].append({
                    'day': date_obj.strftime("%A"),
                    'date': date_obj.strftime("%d %b"),
                    'temp': forecast['temp'],
                    'icon': forecast['icon'],
                    'description': forecast['description']
        })
 

        # if future_forecasts:
        #     for i, forecast in enumerate(future_forecasts, 1):
        #         # Extract time as HH:MM
        #         context[f'time{i}'] = forecast['time'][11:16]
        #         context[f'temp{i}'] = f"{forecast['temp']}"
        #         context[f'hum{i}'] = f"{forecast['humidity']}"
        #         context[f'weather_condition{i}'] = forecast['description'].capitalize()
        # else:
        #     # fallback if API call fails
        #     for i in range(1, 6):
        #         context[f'time{i}'] = "-"
        #         context[f'temp{i}'] = "-"
        #         context[f'hum{i}'] = "-"
        #         context[f'weather_condition{i}'] = "-"

        for i in range(5):
            context[f'time{i+1}'] = future_times[i]
            context[f'temp{i+1}'] = f"{round(future_temp[i], 1)}"
            context[f'hum{i+1}'] = f"{round(future_humidity[i], 1)}"

        return render(request, 'weather.html', context)

    return render(request, 'weather.html')