import requests
import json

url = "http://127.0.0.1:5000/api/simulate"
payload = {
    "deck1": "_random_",
    "deck2": "_random_"
}
headers = {
    "Content-Type": "application/json"
}

print("Sending request to simulate with random decks...")
response = requests.post(url, data=json.dumps(payload), headers=headers)
print("Status Code:", response.status_code)
try:
    res_data = response.json()
    print("Success:", res_data.get("success"))
    if res_data.get("success"):
        vis = res_data.get("visualize", [])
        print("Visualize data length (steps):", len(vis))
        if len(vis) > 0:
            print("First step structure keys/elements:", list(vis[0].keys()) if isinstance(vis[0], dict) else type(vis[0]))
            # Let's inspect the setup or initialization if present
            print("Successfully verified simulation payload generation!")
    else:
        print("Error details:", res_data.get("error"))
except Exception as e:
    print("Failed to parse JSON response:", e)
    print("Response text:", response.text[:500])
