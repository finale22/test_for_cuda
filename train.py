from keras.callbacks import ModelCheckpoint
from model import build_model
from data_preprocessing import preprocess_data, generate
from graph import plot_training_history

def train_model():
    data = preprocess_data()
    model = build_model(data)
    train_gen, val_gen, _, val_steps, _ = generate()

    checkpoint = ModelCheckpoint("models/best_model.h5", save_best_only=True)
    hist = model.fit_generator(train_gen, steps_per_epoch=500, epochs=5, validation_data=val_gen, validation_steps=val_steps, callbacks=[checkpoint])
    
    return hist

if __name__ == "__main__":
    history = train_model()
    plot_training_history(history)