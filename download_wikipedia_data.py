import json
import wikipedia

with open('Common_Medical_Conditions.json', 'r') as f:
    data=json.load(f)

for row in data:
    disease_name = row['disease_name']
    print(disease_name, len(wikipedia.page(f"disease {disease_name}").content))
