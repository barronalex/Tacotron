import os

files = os.listdir('.')
for f in files:
    v = f.find('vol.')
    if v == -1: continue
    new = f[:v] + f[v+6:]
    os.rename(f, new)
