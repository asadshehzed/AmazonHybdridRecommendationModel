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
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_features = tfidf.fit_transform(df['text'].fillna("")).toarray()

    # BERT embeddings
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    model.eval()

    texts = df['text'].fillna("").tolist()[:2000]  # limit for demo
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    bert_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return np.hstack([tfidf_features[:2000], bert_features])

if __name__ == "__main__":
    df = load_gzipped_json("Electronics.jsonl.gz", 5000)
    df = preprocess_data(df)
    X = extract_features(df)
    y = df['rating'].values[:2000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("Hybrid RMSE:", rmse)