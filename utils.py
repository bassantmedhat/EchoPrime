data_split = {
    'train': './data/AS_train.csv',
    'test': './data/AS_test.csv',
    'val': './data/AS_val.csv'
}

# Define label mapping
label_mapping = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def get_split_file(split):
    return data_split.get(split)
