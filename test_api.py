import requests

url = "http://localhost:8000/predict"

data = {
    "Task_Completion": 85,
    "Consistency": 80,
    "Engagement": 90
}

print("Input:", data)
response = requests.post(url, json=data)
print("Prediction:", response.json())