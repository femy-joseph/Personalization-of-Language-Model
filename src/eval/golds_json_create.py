import json

data = {
  "task": "LaMP_4",
  "golds": [
  ]
}

with open("golds_json.json", "w") as f:
  json.dump(data, f)
