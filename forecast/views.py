from django.shortcuts import render
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from timezonefinder import TimezoneFinder
from pytz import timezone, utc

# ------------------- API Setup -------------------
API_KEY = '6ee2176d038f89416fcc2641262f93fb'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

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
            'lat': data['coord']['lat'],
            'lon': data['coord']['lon'],
        }
    except (requests.RequestException, KeyError) as e:
        print(f"Error fetching current weather: {e}")
        return None

def get_5_day_forecast(city):
    try:
        url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        forecasts = []
        for entry in data['list']:
            if "12:00:00" in entry['dt_txt']:
                forecasts.append({
                    'date': entry['dt_txt'].split()[0],
                    'temp': round(entry['main']['temp']),
                    'description': entry['weather'][0]['description'].capitalize(),
                    'icon': entry['weather'][0]['icon']
                })
                if len(forecasts) == 5:
                    break
        return forecasts
    except Exception as e:
        print(f"Error in 5-day forecast: {e}")
        return None

# ------------------- Utility -------------------
def get_compass_direction(degree):
    degree = degree % 360
    compass = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    return next(label for label, start, end in compass if start <= degree < end)

def get_future_hours(lat, lon):
    tf = TimezoneFinder()
    tz_str = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
    local_tz = timezone(tz_str)
    now_utc = datetime.utcnow().replace(tzinfo=utc)
    now_local = now_utc.astimezone(local_tz)
    return [(now_local + timedelta(hours=i + 1)).strftime("%I:%M %p") for i in range(5)]

# ------------------- Machine Learning Setup -------------------
dataset_path = r'C:\Users\ASUS\Downloads\Weather-Forecast-Website-1\weather.csv'
full_data = pd.read_csv(dataset_path)

full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
full_data.dropna(subset=['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp', 'RainTomorrow'], inplace=True)

wind_encoder = LabelEncoder()
full_data['WindGustDir'] = wind_encoder.fit_transform(full_data['WindGustDir'])

rain_encoder = LabelEncoder()
full_data['RainTomorrow'] = rain_encoder.fit_transform(full_data['RainTomorrow'])

features = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']
X = full_data[features]
y = full_data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_train, y_train)

def predict_rain(data_row):
    encoded = rain_model.predict(data_row)[0]
    return rain_encoder.inverse_transform([encoded])[0]

def prepare_regression_data(column):
    X, y = [], []
    for i in range(len(full_data) - 1):
        X.append(full_data[column].iloc[i])
        y.append(full_data[column].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_and_predict_series(current_value, column):
    X, y = prepare_regression_data(column)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions = []
    val = current_value
    for _ in range(5):
        val = model.predict(np.array([[val]]))[0]
        predictions.append(val)
    return predictions

# ------------------- View -------------------
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)
        daily_forecasts = get_5_day_forecast(city)

        if not current_weather:
            return render(request, 'weather.html', {'error': 'Gagal mengambil data cuaca.'})

        compass_dir = get_compass_direction(current_weather['wind_gust_dir'])
        compass_encoded = (
            wind_encoder.transform([compass_dir])[0]
            if compass_dir in wind_encoder.classes_
            else -1
        )

        input_df = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_encoded,
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
        }])

        weather_condition = predict_rain(input_df)
        future_temp = train_and_predict_series(current_weather['temp_min'], 'Temp')
        future_hum = train_and_predict_series(current_weather['humidity'], 'Humidity')
        future_times = get_future_hours(current_weather['lat'], current_weather['lon'])

        context = {
            'location': city,
            'city': current_weather['city'],
            'country': current_weather['country'],
            'current_temp': current_weather['current_temp'],
            'feels_like': current_weather['feels_like'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'wind': current_weather['wind_gust_speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            'weather_condition': weather_condition,
            'time': datetime.utcnow().strftime("%H:%M UTC"),
            #'time': now_local.strftime("%H:%M %Z"),

            
            'date': datetime.utcnow().strftime("%B %d, %Y"),
            'daily_forecasts': [],
            'hourly_forecasts': [],
        }

        if daily_forecasts:
            for forecast in daily_forecasts:
                date_obj = datetime.strptime(forecast['date'], "%Y-%m-%d")
                context['daily_forecasts'].append({
                    'day': date_obj.strftime("%A"),
                    'date': date_obj.strftime("%d %b"),
                    'temp': forecast['temp'],
                    'icon': forecast['icon'],
                    'description': forecast['description'],
                })

        for i in range(5):
            context['hourly_forecasts'].append({
                'time': future_times[i],
                'temp': round(future_temp[i], 1),
                'hum': round(future_hum[i], 1),
            })

        return render(request, 'weather.html', context)

    return render(request, 'weather.html')
