# Amazon Hybrid Recommendation Model

A sophisticated hybrid recommendation system that combines collaborative filtering, content-based filtering, and temporal features to provide personalized product recommendations for Amazon product categories.

## Overview

This project implements a state-of-the-art hybrid recommendation system that leverages multiple machine learning approaches to generate accurate product recommendations. The system combines:

- **Collaborative Filtering**: User-item interaction patterns using Alternating Least Squares (ALS)
- **Content-Based Filtering**: Product features and review text analysis using TF-IDF and BERT embeddings
- **Temporal Features**: Time-based patterns and user behavior evolution
- **Hybrid Integration**: XGBoost-based ensemble model that combines all feature types

## Features

- **Multi-Category Support**: Electronics, Books, Clothing & Jewelry, Home & Kitchen
- **Scalable Architecture**: Configurable sample sizes for experimentation
- **GPU Acceleration**: CUDA support for BERT embeddings and XGBoost training
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG, RMSE, and MAE metrics
- **Business Impact Analysis**: Revenue lift and conversion rate simulations
- **Cold-Start Handling**: Popular item recommendations for new users
- **Automated Visualization**: EDA plots, feature importance, and performance comparisons

## Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for BERT acceleration)
- 8GB+ RAM recommended
- Multi-core CPU for parallel processing

### Python Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
transformers>=4.11.0
xgboost>=1.5.0
implicit>=0.6.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.4.0
wordcloud>=1.8.0
scipy>=1.7.0
tqdm>=4.62.0
joblib>=1.1.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AmazonHybridRecommendationModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required data files:
   - Place your Amazon review data files in the project directory
   - Expected format: `{category}.jsonl.gz` and `meta_{category}.jsonl.gz`

## Data Format

### Review Data (`{category}.jsonl.gz`)
```json
{
  "user_id": "string",
  "parent_asin": "string", 
  "rating": float,
  "text": "string",
  "timestamp": long
}
```

### Metadata (`meta_{category}.jsonl.gz`)
```json
{
  "parent_asin": "string",
  "title": "string",
  "features": ["string"],
  "description": ["string"]
}
```

## Usage

### Basic Execution
```bash
python recsys.py
```

### Configuration
Modify the configuration section in `recsys.py`:
```python
CATEGORIES = ["Electronics", "Books", "Clothing_Shoes_and_Jewelry", "Home_and_Kitchen"]
SAMPLE_SIZES = [50000, 100000]  # Adjust based on your data size
```

### Custom Experiments
```python
from recsys import run_experiment

# Run single experiment
results = run_experiment("Electronics", 50000)
print(f"RMSE: {results['metrics']['rmse']:.4f}")
```

## Model Architecture

### Feature Engineering Pipeline
1. **Text Processing**: TF-IDF vectorization and BERT embeddings
2. **Collaborative Features**: ALS user/item latent factors
3. **Temporal Features**: Review dates, seasonal patterns, user activity evolution
4. **Metadata Integration**: Product titles, features, and descriptions

### Training Process
1. **Data Split**: 70% train, 15% validation, 15% test
2. **Hyperparameter Tuning**: Grid search over XGBoost parameters
3. **Model Training**: XGBoost with early stopping
4. **Evaluation**: Multiple metrics and business impact simulation

### Model Comparison
- **Hybrid Model**: Full feature integration
- **Collaborative Filtering**: User-item interaction patterns only
- **Content-Based**: Text and metadata features only
- **Temporal**: Time-based features only

## Output and Results

### Generated Files
- `experiment_report.csv`: Comprehensive results summary
- `eda_plots/`: Exploratory data analysis visualizations
- `models/`: Trained model files
- `*.png`: Performance comparison charts
- `recsys_log_*.log`: Execution logs

### Key Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Precision@5**: Top-5 recommendation accuracy
- **Recall@5**: Coverage of relevant items
- **NDCG@5**: Normalized Discounted Cumulative Gain

### Business Impact
- **Revenue Lift**: Estimated improvement in conversion rates
- **Sales Lift**: Projected increase in sales performance
- **Conversion Rate**: Enhanced user engagement metrics

## Performance Optimization

### GPU Acceleration
- BERT embeddings generation
- XGBoost training with `gpu_hist` method
- ALS collaborative filtering

### Memory Management
- Configurable batch sizes for BERT processing
- Sparse matrix operations for large datasets
- Efficient feature storage and retrieval

### Scalability
- Parallel processing for multiple categories
- Configurable sample sizes for experimentation
- Modular architecture for easy scaling

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce BERT batch size or sample size
2. **Data Loading Errors**: Verify file paths and data format
3. **Feature Extraction Failures**: Check data quality and missing values

### Performance Tips
- Use smaller sample sizes for initial testing
- Disable BERT if GPU memory is limited
- Adjust XGBoost parameters for faster training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- Amazon review dataset
- Hugging Face Transformers library
- XGBoost development team
- Implicit library for collaborative filtering

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This system is designed for research and educational purposes. Ensure compliance with data usage policies when working with Amazon data.
