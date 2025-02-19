import numpy as np

def load_data():
    fname = "data/jena_climate_2009_2016.csv"
    with open(fname, "r") as f:
        data = f.read()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    # numpy array 변환
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    
    return float_data