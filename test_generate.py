import requests

print("Sending request to /generate...")

response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "Give me a short motivational quote about learning AI."},
)

print("Status:", response.status_code)
print("Response:", response.json())
