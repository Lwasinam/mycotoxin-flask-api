import requests

url = "http://127.0.0.1:5000/predict"
datas = {
  "data": [
    {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 1,
      "Temperature (°C)": 28.5,
      "Humidity (%)": 65.3,
      "Rainfall (mm)": 12.5,
      "Sunlight (hrs/day)": 7.2
    },
    {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 2,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 3,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 4,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 5,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 6,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 7,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    }, 
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 8,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 9,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    },
       {
      "Zone": "Sahel",
      "Year": 2024,
      "Month": 10,
      "Temperature (°C)": 29.1,
      "Humidity (%)": 68.7,
      "Rainfall (mm)": 15.2,
      "Sunlight (hrs/day)": 6.8
    }
 
  ]
}

response = requests.post(url, json=datas)
print(response.json())  # or response.text
