import argparse
from train import train_model
from evaluate import evaluate_model
from graph import plot_training_history

def main():
    parser = argparse.ArgumentParser(description="Time Series Climate Prediction")
    parser.add_argument("--mode", 
                        type=str, required=True, 
                        choices=["train", "evaluate"], 
                        help="Choose an operation mode: train, evaluate")
    
    args = parser.parse_args()

    if args.mode == "train":
        print("Training the model...")
        plot_training_history(train_model())
    elif args.mode == "evaluate":
        print("Evaluating the model...")
        evaluate_model()

if __name__ == "__main__":
    main()