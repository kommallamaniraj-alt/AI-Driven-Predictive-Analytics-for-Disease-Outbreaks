import pandas as pd
import numpy as np

'''
dataset = pd.read_csv("DONs2019.csv", encoding='iso-8859-1')
outbreak = dataset['Outbreak']
country = []
disease = []

for i in range(len(outbreak)):
    data = outbreak[i]
    if '-' in data:
        data = data.replace('-', '\x96')
    arr = data.split("\x96")
    print(arr)
    disease.append(arr[0].strip())
    country.append(arr[1].strip())

dataset = dataset[['Date','Link','Description']]
dataset['Country'] = country
dataset['Outbreak'] = disease

dataset.to_csv("Outbreak.csv", index=False)
'''

dataset = pd.read_csv("Dataset/Outbreak.csv", encoding='iso-8859-1', usecols=['Description'])
dataset = dataset.values
output = ""
for i in range(len(dataset)):
    words = str(dataset[i,0])
    arr = words.split(" ")
    if len(arr) > 10:
        words = words[0:200]
        words = words.replace("\n"," ")
        output += words.strip()+"\n"
print(output)
output = output.encode()
with open("Dataset/testData.csv", "wb") as file:
    file.write(output)
file.close()


country = dataset['Country']
temp = []
hum = []

import pyowm

# key to get climate data
owm = pyowm.OWM('bd5e378503939ddaee76f12ad7a97608')

# Get weather manager
mgr = owm.weather_manager()

for i in range(len(country)):
    try:
        place = country[i]
        observation = mgr.weather_at_place(place)#get weather data for given country
        weather = observation.weather
        temperature = weather.temperature('celsius')['temp']
        humidity = weather.humidity
        temp.append(temperature)
        hum.append(humidity)
        print(place+" "+str(temperature)+" "+str(humidity))
    except:
        temperature = np.nan
        humidity = np.nan
        temp.append(temperature)
        hum.append(humidity)
        print(place+" "+str(temperature)+" "+str(humidity))
        pass

data = dataset[['Date','Link','Description', 'Country', 'Outbreak']]
data['Temperature'] = temp
data['Humidity'] = hum

data.to_csv("data.csv", index=False)




