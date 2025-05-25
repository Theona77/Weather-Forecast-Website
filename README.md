Instruksi menjalankan filenya!

pada views.py : ganti dulu read file csv seusai path di komputer masing"

Jika sudah, Buka terminal
Opsi 1: 
python manage.py runserver

Opsi 2: (Jika tidak ada env maka aktivasi enviromentnya dulu)
.\myenv\Scripts\Activate.ps1
cd .\Weather-Forecast-Website-1
python manage.py runserver

Jika sudah, 
Open http://127.0.0.1:8000/ di browser
Masukin nama kota maka output akan muncul

Notes:
Pastikan ada internet karena data akan diambil dari API OpenWeather secara realtime
