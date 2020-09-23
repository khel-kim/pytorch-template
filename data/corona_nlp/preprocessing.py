import os
import sys

if __name__ == "__main__":
    __file_path = os.path.abspath(__file__)
    sys.path.append("/".join(__file_path.split("/")[:-3]))

    from sklearn.model_selection import train_test_split
    from utils.utils import set_random_seed, read_csv, save_json

    _data_root = 'data/corona_nlp'
    _train_name = 'origin/Corona_NLP_train.csv'
    _test_name = 'origin/Corona_NLP_test.csv'
    _seed = 0

    set_random_seed(_seed)

    train_path = f"{_data_root}/{_train_name}"
    test_path = f"{_data_root}/{_test_name}"

    train = read_csv(train_path, encoding='latin1')
    test = read_csv(test_path, encoding='latin1')
    train, dev = train_test_split(train, test_size=0.1)  # hard coding

    cat2idx = {
        'Extremely Negative': 0,
        'Negative': 1,
        'Neutral': 2,
        'Positive': 3,
        'Extremely Positive': 4,
    }  # hard coding

    train_dic, dev_dic, test_dic = {}, {}, {}

    train_dic["texts"] = list(train["OriginalTweet"])
    train_dic["categories"] = [cat2idx[cat] for cat in train['Sentiment']]

    dev_dic["texts"] = list(dev["OriginalTweet"])
    dev_dic["categories"] = [cat2idx[cat] for cat in dev['Sentiment']]

    test_dic["texts"] = list(test["OriginalTweet"])
    test_dic["categories"] = [cat2idx[cat] for cat in test['Sentiment']]

    train_idx = list(train.index)
    dev_idx = list(dev.index)

    try:
        os.mkdir(f"{_data_root}/info")
    except FileExistsError:
        pass

    save_json(f"{_data_root}/train.json", train_dic)
    save_json(f"{_data_root}/dev.json", dev_dic)
    save_json(f"{_data_root}/test.json", test_dic)
    save_json(f"{_data_root}/info/cat2idx.json", cat2idx)
    save_json(f"{_data_root}/info/train_idx.json", train_idx)
    save_json(f"{_data_root}/info/dev_idx.json", dev_idx)
