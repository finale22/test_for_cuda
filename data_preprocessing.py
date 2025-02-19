import numpy as np
from data_loader import load_data

def preprocess_data():
    data = load_data()
    
    # 정규화
    mean = data[:200000].mean(axis=0)
    data = data - mean
    std = data[:200000].std(axis=0)
    data = data / std
    return data

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
    
def generate():
    lookback, delay, batch_size = 1440, 144, 128
    data = preprocess_data()
    train_gen = generator(data, lookback, delay, 0, 200000, shuffle=True)
    val_gen = generator(data, lookback, delay, 200001, 300000)
    test_gen = generator(data, lookback, delay, 300001, None)

    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(data) - 300001 - lookback) // batch_size
    return train_gen, val_gen, test_gen, val_steps, test_steps