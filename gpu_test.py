import pandas as pd

df = pd.read_csv("data/rsrc.csv", encoding="latin1")

print(df.columns.tolist())
