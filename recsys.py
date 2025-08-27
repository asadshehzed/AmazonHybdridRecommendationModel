import gzip
import json
import os
import time
import warnings
import pandas as pd
import numpy as np
import implicit
import torch
import xgboost as xgb
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from scipy.sparse import coo_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import joblib
from datetime import datetime
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CATEGORIES = ["Electronics", "Books", "Clothing_Shoes_and_Jewelry", "Home_and_Kitchen"]
SAMPLE_SIZES = [50000, 100000]  
EXPERIMENT_RESULTS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CORES = os.cpu_count()
MODEL_COMPARISONS = {}

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"Using {NUM_CORES} CPU cores")

# Setup logging
def setup_logger():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f"recsys_log_{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return log_file

LOGFILE = setup_logger()

def log_message(message, level=logging.INFO):
    if level == logging.INFO:
        logging.info(message)
    elif level == logging.WARNING:
        logging.warning(message)
    elif level == logging.ERROR:
        logging.error(message)
    elif level == logging.CRITICAL:
        logging.critical(message)
    print(message)

# Data Loading
def load_gzipped_json(file_path, sample_size=None):
    """Load gzipped JSONL file with progress tracking"""
    log_message(f"Loading {file_path}...")
    data = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        log_message(f"Loaded {len(data)} records from {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        log_message(f"Error loading {file_path}: {str(e)}", logging.ERROR)
        return pd.DataFrame()

# Data Preprocessing with Temporal Features
def preprocess_data(reviews_df, meta_df=None):
    """Clean and prepare data with temporal features"""
    log_message("Preprocessing data...")
    
    # Initial cleaning
    reviews_df = reviews_df.dropna(subset=['rating', 'text'])
    reviews_df = reviews_df[reviews_df['rating'] >= 1]
    
    # Handle timestamp conversion safely
    if 'timestamp' in reviews_df.columns:
        # Convert milliseconds to seconds
        reviews_df['timestamp'] = reviews_df['timestamp'] / 1000
        
        # Filter out invalid timestamps
        min_valid_ts = pd.Timestamp('1995-01-01').timestamp()
        max_valid_ts = pd.Timestamp('2025-12-31').timestamp()
        valid_ts = reviews_df['timestamp'].between(min_valid_ts, max_valid_ts)
        reviews_df = reviews_df[valid_ts]
        
        # Convert to datetime
        reviews_df['review_date'] = pd.to_datetime(reviews_df['timestamp'], unit='s', errors='coerce')
    else:
        reviews_df['review_date'] = pd.Timestamp.now()
    
    # Filter invalid dates
    min_date = pd.Timestamp('2000-01-01')
    max_date = pd.Timestamp.today()
    reviews_df = reviews_df[
        (reviews_df['review_date'] >= min_date) & 
        (reviews_df['review_date'] <= max_date)
    ]
    
    # Extract temporal features
    reviews_df['review_year'] = reviews_df['review_date'].dt.year
    reviews_df['review_month'] = reviews_df['review_date'].dt.month
    reviews_df['review_day'] = reviews_df['review_date'].dt.day
    reviews_df['review_dayofweek'] = reviews_df['review_date'].dt.dayofweek
    
    # Calculate days since first review
    reviews_df['days_since_first'] = (reviews_df['review_date'] - reviews_df.groupby('user_id')['review_date'].transform('min')).dt.days
    
    # Text cleaning
    reviews_df['clean_text'] = reviews_df['text'].str.lower().str.replace(r'<.*?>', '', regex=True)
    reviews_df['review_length'] = reviews_df['text'].apply(len)
    
    # Metadata merge
    if meta_df is not None and not meta_df.empty:
        log_message("Merging with metadata...")
        
        # Create text representation of metadata
        def create_meta_text(row):
            parts = []
            if 'title' in row and pd.notna(row['title']):
                parts.append(str(row['title']))
            if 'features' in row and isinstance(row['features'], list):
                parts.append(" ".join(row['features']))
            if 'description' in row and isinstance(row['description'], list):
                parts.append(" ".join(row['description']))
            return " ".join(parts)
        
        meta_df['meta_text'] = meta_df.apply(create_meta_text, axis=1)
        meta_subset = meta_df[['parent_asin', 'meta_text']].drop_duplicates(subset=['parent_asin'])
        
        # Merge with reviews
        reviews_df = reviews_df.merge(
            meta_subset, 
            on='parent_asin', 
            how='left'
        )
        
        # Combine text features (use only metadata for prediction)
        reviews_df['full_text'] = reviews_df['meta_text'].fillna('')
    else:
        log_message("No metadata available, using review text only", logging.WARNING)
        reviews_df['full_text'] = reviews_df['clean_text']
    
    return reviews_df[['user_id', 'parent_asin', 'rating', 'full_text', 'review_length',
                      'review_date', 'review_year', 'review_month', 'review_day', 
                      'review_dayofweek', 'days_since_first']]


def perform_eda(df, category, sample_size):
    """Generate comprehensive EDA visualizations"""
    os.makedirs("eda_plots", exist_ok=True)
    
    try:
        # 1. Rating Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rating', data=df, palette='viridis')
        plt.title(f'Rating Distribution - {category} ({sample_size} samples)')
        plt.savefig(f'eda_plots/ratings_{category}_{sample_size}.png', dpi=300)
        plt.close()

        # 2. Temporal Trends
        plt.figure(figsize=(12, 6))
        df.set_index('review_date').resample('M')['rating'].count().plot()
        plt.title(f'Reviews Over Time - {category}')
        plt.ylabel('Number of Reviews')
        plt.savefig(f'eda_plots/temporal_{category}_{sample_size}.png', dpi=300)
        plt.close()

        # 3. Review Length Analysis
        plt.figure(figsize=(10, 6))
        sns.histplot(df['review_length'], bins=50, kde=True)
        plt.title(f'Review Length Distribution - {category}')
        plt.xlabel('Character Count')
        plt.savefig(f'eda_plots/length_{category}_{sample_size}.png', dpi=300)
        plt.close()

        # 4. Word Cloud
        plt.figure(figsize=(12, 8))
        text = " ".join(review for review in df['full_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Common Words - {category}')
        plt.savefig(f'eda_plots/wordcloud_{category}_{sample_size}.png', dpi=300)
        plt.close()

        # 5. User Activity
        plt.figure(figsize=(10, 6))
        user_review_counts = df['user_id'].value_counts()
        sns.histplot(user_review_counts, bins=50, log_scale=(False, True))
        plt.title(f'User Activity Distribution - {category}')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Log Count of Users')
        plt.savefig(f'eda_plots/user_activity_{category}_{sample_size}.png', dpi=300)
        plt.close()

    except Exception as e:
        log_message(f"EDA failed: {str(e)}", logging.ERROR)

# Feature Engineering with BERT and Collaborative Filtering
def extract_features(df, use_bert=True):
    """Generate features with GPU support"""
    features = {}
    
    # TF-IDF Features
    log_message("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    
    # Handle empty text cases
    df['full_text'] = df['full_text'].fillna('')
    if df['full_text'].empty or df['full_text'].isnull().all():
        tfidf_features = coo_matrix((len(df), 500))
    else:
        tfidf_features = tfidf.fit_transform(df['full_text'])
    
    features['tfidf'] = tfidf_features
    
    # BERT Embeddings
    if use_bert and DEVICE == "cuda":
        try:
            log_message("Generating BERT embeddings...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
            model.eval()
            
            def process_batch(texts):
                inputs = tokenizer(
                    texts, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=128, 
                    padding='max_length'
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            batch_size = 128  # Reduced for GPU memory safety
            bert_features = []
            
            for i in tqdm(range(0, len(df), batch_size), desc="BERT Processing"):
                batch_texts = df['full_text'].iloc[i:i+batch_size].fillna('').tolist()
                batch_embeddings = process_batch(batch_texts)
                bert_features.append(batch_embeddings)
            
            features['bert'] = np.vstack(bert_features)
        except Exception as e:
            log_message(f"BERT failed: {str(e)}", logging.ERROR)
            features['bert'] = np.zeros((len(df), 768))
    else:
        features['bert'] = np.zeros((len(df), 768))
    
    # Collaborative Filtering
    log_message("Training collaborative filtering model...")
    try:
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        user_ids = user_encoder.fit_transform(df['user_id'])
        item_ids = item_encoder.fit_transform(df['parent_asin'])
        ratings = df['rating'].values
        
        user_item_matrix = coo_matrix((ratings, (user_ids, item_ids)))
        
        als_model = implicit.als.AlternatingLeastSquares(
            factors=64,  # Reduced for memory efficiency
            regularization=0.1,
            iterations=15,
            use_gpu=(DEVICE=="cuda")
        )
        als_model.fit(user_item_matrix)
        
        features['user_factors'] = als_model.user_factors[user_ids]
        features['item_factors'] = als_model.item_factors[item_ids]
        features['user_encoder'] = user_encoder
        features['item_encoder'] = item_encoder
        features['als_model'] = als_model
    except Exception as e:
        log_message(f"Collaborative filtering failed: {str(e)}", logging.ERROR)
        features['user_factors'] = np.zeros((len(df), 64))
        features['item_factors'] = np.zeros((len(df), 64))
        features['user_encoder'] = None
        features['item_encoder'] = None
        features['als_model'] = None
    
    # Temporal features
    temporal_features = ['review_year', 'review_month', 'review_day', 
                         'review_dayofweek', 'days_since_first']
    for feature in temporal_features:
        if feature in df.columns:
            features[feature] = df[feature].values.reshape(-1, 1)
        else:
            features[feature] = np.zeros((len(df), 1))
    
    # Review length feature
    features['review_length'] = df['review_length'].values.reshape(-1, 1)
    
    return features

# Hyperparameter Tuning
def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """Optimize hyperparameters using basic search"""
    log_message("Using basic hyperparameter tuning")
    best_rmse = float('inf')
    best_params = {}
    
    param_combinations = [
        {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6},
        {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 8},
        {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 7},
        {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 5}
    ]
    
    for params in param_combinations:
        log_message(f"Testing params: {params}")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='gpu_hist' if DEVICE == "cuda" else 'auto',
            **params
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            
        log_message(f"RMSE: {rmse:.4f} with params {params}")
    
    log_message(f"Best RMSE: {best_rmse:.4f} with params {best_params}")
    return best_params

# Hybrid Model Training with Hyperparameter Tuning
def train_hybrid_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate hybrid model with hyperparameter tuning"""
    # Hyperparameter optimization
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Train final model with best parameters
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    final_model = xgb.train(
        {**best_params, **{'objective': 'reg:squarederror', 'eval_metric': 'rmse'}}, 
        dtrain,
        num_boost_round=best_params.get('n_estimators', 500),
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Evaluate on test set
    preds = final_model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    return final_model, rmse, mae, best_params

# Advanced Recommendation Evaluation
def evaluate_recommendations(model, df, X_test, user_encoder, item_encoder, als_model, top_k=5):
    """Evaluate recommendations using precision, recall, and NDCG"""
    # Select test users
    test_users = df['user_id'].unique()[:500]  # Reduced for speed
    
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    for user in tqdm(test_users, desc="Evaluating Recommendations"):
        # Get actual items the user has rated
        user_mask = (df['user_id'] == user)
        actual_items = set(df.loc[user_mask, 'parent_asin'])
        if not actual_items: 
            continue
            
        # Get user features
        if user_mask.sum() == 0:
            continue
            
        user_data = X_test[user_mask.values]
        
        # Predict ratings
        try:
            pred_scores = model.predict(xgb.DMatrix(user_data))
        except:
            continue
            
        # Get top K recommendations
        top_indices = np.argsort(pred_scores)[-top_k:]
        top_items = df.loc[user_mask].iloc[top_indices]['parent_asin'].tolist()
        recommended_set = set(top_items)
        
        # Calculate precision and recall
        relevant_count = len(recommended_set & actual_items)
        precision = relevant_count / top_k
        recall = relevant_count / len(actual_items) if actual_items else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        
        # Calculate NDCG
        relevance_scores = [1 if item in actual_items else 0 for item in top_items]
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(top_k, len(actual_items)))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    
    return {
        'precision@5': avg_precision,
        'recall@5': avg_recall,
        'ndcg@5': avg_ndcg
    }

# Business Impact Simulation
def simulate_business_impact(eval_results, avg_price=25.0):
    """Simulate business impact based on evaluation metrics"""
    baseline_conversion = 0.05  # 5% conversion rate baseline
    precision_gain = eval_results['precision@5']
    
    # Improved conversion based on precision gain
    improved_conversion = min(baseline_conversion * (1 + precision_gain * 0.7), 0.3)
    revenue_lift = (improved_conversion - baseline_conversion) / baseline_conversion
    
    # Estimate sales lift
    estimated_sales_lift = revenue_lift * 100  # Percentage
    
    return {
        'revenue_lift': revenue_lift,
        'estimated_sales_lift': estimated_sales_lift,
        'improved_conversion_rate': improved_conversion
    }

# Model Comparison
def compare_models(X_train, y_train, X_test, y_test):
    """Compare different modeling approaches"""
    results = {}
    metrics = {}
    
    # 1. Collaborative Filtering Only
    log_message("Training Collaborative Filtering only model...")
    try:
        cf_features = np.hstack([X_train[:, :128]])  # First 64 user + 64 item factors
        cf_model = xgb.XGBRegressor(
            n_estimators=300,
            tree_method='gpu_hist' if DEVICE == "cuda" else 'auto'
        )
        cf_model.fit(cf_features, y_train)
        cf_preds = cf_model.predict(np.hstack([X_test[:, :128]]))
        results['collab_filter'] = cf_model
        metrics['collab_filter'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, cf_preds)),
            'mae': mean_absolute_error(y_test, cf_preds)
        }
    except Exception as e:
        log_message(f"Collaborative filter failed: {str(e)}", logging.ERROR)
        metrics['collab_filter'] = {'rmse': 0, 'mae': 0}
    
    # 2. Content-Based Only
    log_message("Training Content-Based only model...")
    try:
        # Find indices for content features
        tfidf_end = X_train.shape[1] - 768 - 5
        bert_start = tfidf_end
        
        content_features = np.hstack([
            X_train[:, 128:tfidf_end],  # TF-IDF
            X_train[:, bert_start:bert_start+768]  # BERT
        ])
        
        content_model = xgb.XGBRegressor(
            n_estimators=300,
            tree_method='gpu_hist' if DEVICE == "cuda" else 'auto'
        )
        content_model.fit(content_features, y_train)
        
        test_content = np.hstack([
            X_test[:, 128:tfidf_end],
            X_test[:, bert_start:bert_start+768]
        ])
        
        content_preds = content_model.predict(test_content)
        results['content_based'] = content_model
        metrics['content_based'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, content_preds)),
            'mae': mean_absolute_error(y_test, content_preds)
        }
    except Exception as e:
        log_message(f"Content-based failed: {str(e)}", logging.ERROR)
        metrics['content_based'] = {'rmse': 0, 'mae': 0}
    
    # 3. Temporal Model Only
    log_message("Training Temporal Features only model...")
    try:
        temporal_features = X_train[:, -5:]  # Last 5 features are temporal
        temporal_model = xgb.XGBRegressor(
            n_estimators=300,
            tree_method='gpu_hist' if DEVICE == "cuda" else 'auto'
        )
        temporal_model.fit(temporal_features, y_train)
        temporal_preds = temporal_model.predict(X_test[:, -5:])
        results['temporal'] = temporal_model
        metrics['temporal'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, temporal_preds)),
            'mae': mean_absolute_error(y_test, temporal_preds)
        }
    except Exception as e:
        log_message(f"Temporal model failed: {str(e)}", logging.ERROR)
        metrics['temporal'] = {'rmse': 0, 'mae': 0}
    
    return results, metrics

# Visualization Functions
def plot_feature_importance(model, category, sample_size):
    """Plot feature importance"""
    try:
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=15, importance_type='gain')
        plt.title(f'Feature Importance - {category} {sample_size}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{category}_{sample_size}.png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        log_message(f"Failed to plot feature importance: {str(e)}", logging.ERROR)
        return False

def plot_metric_comparison(experiment_results, metric='rmse'):
    """Plot comparison of metrics across experiments"""
    try:
        plt.figure(figsize=(12, 8))
        categories = []
        values = []
        
        for exp_name, results in experiment_results.items():
            if results is None:
                continue
            categories.append(exp_name)
            values.append(results['metrics'][metric])
        
        plt.bar(categories, values, color='skyblue')
        plt.title(f'{metric.upper()} Comparison Across Experiments')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        log_message(f"Failed to plot metric comparison: {str(e)}", logging.ERROR)
        return False

def plot_metrics_vs_samples(experiment_results):
    """Plot metrics vs sample sizes"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        data = []
        for exp_name, results in experiment_results.items():
            if results is None:
                continue
            parts = exp_name.split('_')
            category = parts[0]
            sample_size = int(parts[1])
            data.append({
                'Category': category,
                'Sample Size': sample_size,
                'RMSE': results['metrics']['rmse'],
                'Precision@5': results['eval_results']['precision@5']
            })
        
        if not data:
            return False
            
        df = pd.DataFrame(data)
        
        # Plot RMSE
        plt.subplot(2, 1, 1)
        for cat in df['Category'].unique():
            subset = df[df['Category'] == cat]
            plt.plot(subset['Sample Size'], subset['RMSE'], marker='o', label=cat)
        plt.legend()
        plt.title('RMSE vs Sample Size')
        plt.grid(True)
        
        # Plot Precision
        plt.subplot(2, 1, 2)
        for cat in df['Category'].unique():
            subset = df[df['Category'] == cat]
            plt.plot(subset['Sample Size'], subset['Precision@5'], marker='o', label=cat)
        plt.legend()
        plt.title('Precision@5 vs Sample Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('metrics_vs_samples.png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        log_message(f"Failed to plot metrics vs samples: {str(e)}", logging.ERROR)
        return False

def plot_model_comparison(experiment_results):
    """Create detailed model comparison visualizations"""
    # Prepare comparison data
    comparison_data = []
    
    for exp_name, results in experiment_results.items():
        if results is None: 
            continue
            
        # Hybrid model results
        comparison_data.append({
            'Experiment': exp_name,
            'Model': 'Hybrid',
            'RMSE': results['metrics']['rmse'],
            'Precision@5': results['eval_results']['precision@5']
        })
        
        # Baseline models
        for model_type, metrics in results['comparison_metrics'].items():
            comparison_data.append({
                'Experiment': exp_name,
                'Model': model_type.capitalize(),
                'RMSE': metrics['rmse'],
                'Precision@5': None  # Not available for baselines
            })
    
    if not comparison_data:
        log_message("No results available for visualization", logging.WARNING)
        return
    
    df = pd.DataFrame(comparison_data)
    
    # 1. RMSE Comparison
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Experiment', y='RMSE', hue='Model', data=df)
    plt.title('RMSE Comparison Across Models')
    plt.ylabel('RMSE (Lower is Better)')
    plt.xticks(rotation=15)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig('results_rmse_comparison.png', dpi=300)
    plt.close()
    
    # 2. Precision@5 Comparison
    plt.figure(figsize=(14, 8))
    hybrid_df = df[df['Model'] == 'Hybrid'].dropna()
    sns.barplot(x='Experiment', y='Precision@5', data=hybrid_df, palette='Blues_d')
    plt.title('Precision@5 for Hybrid Models')
    plt.ylabel('Precision@5 (Higher is Better)')
    plt.ylim(0, 0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('results_precision_comparison.png', dpi=300)
    plt.close()
    
    # 3. Performance vs Sample Size
    if not hybrid_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Extract sample size from experiment name
        hybrid_df['Sample Size'] = hybrid_df['Experiment'].apply(lambda x: int(x.split('_')[-1]))
        hybrid_df['Category'] = hybrid_df['Experiment'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        
        # RMSE vs Sample Size
        plt.subplot(2, 1, 1)
        sns.lineplot(x='Sample Size', y='RMSE', hue='Category', 
                     data=hybrid_df, marker='o', markersize=8)
        plt.title('RMSE vs Sample Size')
        plt.grid(True)
        
        # Precision vs Sample Size
        plt.subplot(2, 1, 2)
        sns.lineplot(x='Sample Size', y='Precision@5', hue='Category', 
                     data=hybrid_df, marker='o', markersize=8)
        plt.title('Precision@5 vs Sample Size')
        plt.xlabel('Sample Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results_scalability.png', dpi=300)
        plt.close()

# Cold-start User Handling
def handle_cold_start(meta_df, top_k=5):
    """Generate recommendations for cold-start users"""
    try:
        if meta_df is not None and not meta_df.empty:
            # Get most reviewed items
            popular_items = meta_df['parent_asin'].value_counts().index.tolist()
            recommendations = popular_items[:top_k]
        else:
            # Fallback to generic recommendations
            recommendations = ["B0000530HU", "B0000530HX", "B0000530HY", 
                              "B0000530HZ", "B0000530I0"][:top_k]
        
        return {
            'user_id': "cold_start_user",
            'recommendations': recommendations,
            'strategy': 'popular_items'
        }
    except:
        return {
            'user_id': "cold_start_user",
            'recommendations': ["B0000530HU", "B0000530HX", "B0000530HY", 
                               "B0000530HZ", "B0000530I0"][:top_k],
            'strategy': 'fallback_items'
        }

# Main Workflow
def run_experiment(category, sample_size):
    """Run full experiment pipeline"""
    log_message(f"\n{'='*50}")
    log_message(f"Starting Experiment: {category} ({sample_size} samples)")
    log_message(f"{'='*50}")
    
    # Load data
    reviews_file = f"{category}.jsonl.gz"
    meta_file = f"meta_{category}.jsonl.gz"
    
    reviews_df = load_gzipped_json(reviews_file, sample_size)
    meta_df = load_gzipped_json(meta_file, sample_size) if os.path.exists(meta_file) else None
    
    if reviews_df.empty:
        log_message(f"Skipping {category} due to data loading issues", logging.WARNING)
        return None
    
    # Preprocess
    df = preprocess_data(reviews_df, meta_df)
    perform_eda(df, category, sample_size)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    # Feature engineering
    log_message("Extracting features for training set...")
    train_features = extract_features(train_df, use_bert=True)
    
    # Combine features for training
    feature_components = [
        train_features['user_factors'],
        train_features['item_factors'],
        train_features['review_length'],
        train_features['tfidf'].toarray(),
        train_features['bert'],
        train_features['review_year'],
        train_features['review_month'],
        train_features['review_day'],
        train_features['review_dayofweek'],
        train_features['days_since_first']
    ]
    X_train = np.hstack(feature_components)
    y_train = train_df['rating'].values
    
    # Validation set features
    log_message("Extracting features for validation set...")
    val_features = extract_features(val_df, use_bert=False)
    val_components = [
        val_features['user_factors'],
        val_features['item_factors'],
        val_features['review_length'],
        val_features['tfidf'].toarray(),
        val_features['bert'],
        val_features['review_year'],
        val_features['review_month'],
        val_features['review_day'],
        val_features['review_dayofweek'],
        val_features['days_since_first']
    ]
    X_val = np.hstack(val_components)
    y_val = val_df['rating'].values
    
    # Test set features
    log_message("Extracting features for test set...")
    test_features = extract_features(test_df, use_bert=False)
    test_components = [
        test_features['user_factors'],
        test_features['item_factors'],
        test_features['review_length'],
        test_features['tfidf'].toarray(),
        test_features['bert'],
        test_features['review_year'],
        test_features['review_month'],
        test_features['review_day'],
        test_features['review_dayofweek'],
        test_features['days_since_first']
    ]
    X_test = np.hstack(test_components)
    y_test = test_df['rating'].values
    
    # Train hybrid model
    log_message("Training hybrid model...")
    try:
        model, rmse, mae, best_params = train_hybrid_model(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
    except Exception as e:
        log_message(f"Model training failed: {str(e)}", logging.ERROR)
        return None
    
    log_message(f"Hybrid Model RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Evaluate recommendations
    log_message("Evaluating recommendations...")
    try:
        eval_results = evaluate_recommendations(
            model, 
            test_df, 
            X_test,
            train_features.get('user_encoder'),
            train_features.get('item_encoder'),
            train_features.get('als_model')
        )
    except Exception as e:
        log_message(f"Recommendation evaluation failed: {str(e)}", logging.ERROR)
        eval_results = {
            'precision@5': 0,
            'recall@5': 0,
            'ndcg@5': 0
        }
    
    # Business impact simulation
    log_message("Simulating business impact...")
    business_impact = simulate_business_impact(eval_results)
    
    # Compare with other models
    log_message("Comparing with baseline models...")
    try:
        comparison_models, comparison_metrics = compare_models(X_train, y_train, X_test, y_test)
    except Exception as e:
        log_message(f"Model comparison failed: {str(e)}", logging.ERROR)
        comparison_metrics = {}
    
    # Save models
    model_dir = f"models/{category}_{sample_size}"
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        model.save_model(f"{model_dir}/hybrid_model.json")
    except:
        pass
    
    # Save evaluation results
    results = {
        'metrics': {'rmse': rmse, 'mae': mae},
        'eval_results': eval_results,
        'business_impact': business_impact,
        'best_params': best_params,
        'comparison_metrics': comparison_metrics
    }
    
    # Visualization
    plot_feature_importance(model, category, sample_size)
    
    # Cold-start example
    cold_start_rec = handle_cold_start(meta_df)
    log_message(f"Cold-start recommendations: {cold_start_rec}")
    
    log_message(f"Completed experiment: {category} ({sample_size} samples)")
    return results

# Generate Report
def generate_report(experiment_results):
    """Generate CSV and visual report of all experiment results"""
    report_data = []
    for exp, res in experiment_results.items():
        if res is None:
            continue
            
        report_data.append({
            'Experiment': exp,
            'RMSE': res['metrics']['rmse'],
            'MAE': res['metrics']['mae'],
            'Precision@5': res['eval_results']['precision@5'],
            'Recall@5': res['eval_results']['recall@5'],
            'NDCG@5': res['eval_results']['ndcg@5'],
            'Revenue Lift': f"{res['business_impact']['revenue_lift']:.2%}",
            'Sales Lift': f"{res['business_impact']['estimated_sales_lift']:.2f}%",
            'Best Parameters': str(res['best_params'])
        })
        
        # Add comparison models
        for model_type, metrics in res['comparison_metrics'].items():
            report_data.append({
                'Experiment': f"{exp}_{model_type}",
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'Precision@5': '-',
                'Recall@5': '-',
                'NDCG@5': '-',
                'Revenue Lift': '-',
                'Sales Lift': '-',
                'Best Parameters': '-'
            })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv('experiment_report.csv', index=False)
    
    # Create visualizations
    plot_metric_comparison(experiment_results, metric='rmse')
    plot_metric_comparison(experiment_results, metric='Precision@5')
    plot_metrics_vs_samples(experiment_results)
    plot_model_comparison(experiment_results)

    return report_df

# Main Execution
if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run experiments
    for category in CATEGORIES:
        for sample_size in SAMPLE_SIZES:
            try:
                log_message(f"Running experiment: {category} ({sample_size} samples)")
                results = run_experiment(category, sample_size)
                EXPERIMENT_RESULTS[f"{category}_{sample_size}"] = results
            except Exception as e:
                log_message(f"Error in experiment {category}_{sample_size}: {str(e)}", logging.ERROR)
                EXPERIMENT_RESULTS[f"{category}_{sample_size}"] = None
    
    try:
        plot_model_comparison(EXPERIMENT_RESULTS)
    except Exception as e:
        log_message(f"Visualization failed: {str(e)}", logging.ERROR)

    # Generate final report
    log_message("Generating final report...")
    report_df = generate_report(EXPERIMENT_RESULTS)
    log_message("Saved report to experiment_report.csv")
    
    # Print final report
    print("\nFinal Report:")
    print(report_df.to_string(index=False))
    
    # Final memory report
    if DEVICE == "cuda":
        log_message(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e6:.2f} MB allocated")
        log_message(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1e6:.2f} MB")
    
    log_message("All experiments completed!")
    