from main import *

if __name__ == "__main__":
    # Get model name
    model_name = sys.argv[1].lower()

    train(model_name)
    test(model_name)
