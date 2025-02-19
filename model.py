from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import RMSprop

def build_model(data):
    clear_session()
    model = Sequential()
    model.add(GRU(32, input_shape=(None, data.shape[-1])))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    return model
