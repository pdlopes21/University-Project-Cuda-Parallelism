import numpy as np
import pandas as pd


df = pd.read_csv("data.csv")

funs = [ line.strip() for line in open("functions.txt").readlines() ]

def score(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns:
        line = line.replace(f"_{c}_", f"(df[\"{c}\"].values)")
    a = eval(line)
    b = df["y"]
    e = np.square(np.subtract(a, b)).mean()
    return e

l = funs[0]
print(score(l), l)

r = min([ (score(line), line) for line in funs ])
print(f"{r[0]} {r[1]}")