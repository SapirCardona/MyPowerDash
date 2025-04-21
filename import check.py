import pandas as pd

with open(r"C:\Users\sapir\OneDrive\Desktop\data analyst\עצמאי\חשמל 2.0\meter_22011461_LP_02-01-2025.csv", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines[:15]):
    print(f"{i+1}: {line.strip()}")