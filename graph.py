import matplotlib.pyplot as plt

def plot_training_history(hist):
    history = hist.history
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Trainig loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()