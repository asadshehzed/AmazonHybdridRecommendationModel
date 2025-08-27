import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

def perform_eda(df, category):
    # Rating distribution
    sns.countplot(x="rating", data=df)
    plt.title(f"Rating Distribution - {category}")
    plt.show()

    # Review length
    sns.histplot(df['review_length'], bins=50, kde=True)
    plt.title(f"Review Length - {category}")
    plt.show()

if __name__ == "__main__":
    category = "Electronics"
    reviews_file = f"{category}.jsonl.gz"
    
    df = load_gzipped_json(reviews_file, sample_size=10000)
    df = preprocess_data(df)
    perform_eda(df, category)
