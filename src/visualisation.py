def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data