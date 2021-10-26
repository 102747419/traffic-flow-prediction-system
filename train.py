from main import *

if __name__ == "__main__":
    # Get model name
    model_name = sys.argv[1].lower()

    # Load the data
    data = load_data()

    train(data, model_name)
    # test_model()
