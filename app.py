import requests
    
url = 'http://localhost:8000/predict'
response = requests.get(url)
response.json()