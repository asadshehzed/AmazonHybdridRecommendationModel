import gzip, json, warnings
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_gzipped_json(file_path, sample_size=None):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def preprocess_data(df):
    df = df.dropna(subset=['rating', 'text'])
    df['review_length'] = df['text'].apply(len)
    return df[['user_id', 'parent_asin', 'rating', 'text', 'review_length']]

def extract_features(df):
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'].fillna(""))

    # Collaborative Filtering (ALS)
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    user_ids = user_enc.fit_transform(df['user_id'])
    item_ids = item_enc.fit_transform(df['parent_asin'])

    matrix = coo_matrix((df['rating'], (user_ids, item_ids)))
    als = AlternatingLeastSquares(factors=32, iterations=10)
    als.fit(matrix)

    return tfidf_matrix, als, user_enc, item_enc

if __name__ == "__main__":
    df = load_gzipped_json("Electronics.jsonl.gz", 20000)
    df = preprocess_data(df)
    tfidf_matrix, als, user_enc, item_enc = extract_features(df)
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("ALS user factors:", als.user_factors.shape)
