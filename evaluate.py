from sklearn.metrics import mean_absolute_error
from data_preprocessing import preprocess_data, generate
from keras.models import load_model


def evaluate_model():
    data = preprocess_data()
    _, _, test_gen, _, test_steps = generate()
    model = load_model("models/best_model.h5")

    y_pred = model.predict_generator(test_gen, steps=test_steps)
    y_test = data[:, 1][300001:300001+len(y_pred)]

    print(f"GRU: {mean_absolute_error(y_test, y_pred)}")

if __name__ == "__main__":
    evaluate_model()
