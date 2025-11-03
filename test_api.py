import requests

print("Sending request...")

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "What is LangChain?"}
)

print("Status:", response.status_code)
print("Response:", response.json())
