# TED Talk Popularity Prediction Using NLP

This project uses **Natural Language Processing (NLP)** to predict the popularity of TED Talks by classifying them into "High Views" or "Low Views" categories. The goal is to understand what factors contribute to high viewership and to leverage machine learning models for classification.

## Objective

- Classify TED Talks as "High Views" or "Low Views" based on their transcript and metadata.
- Use text preprocessing techniques and machine learning models to enhance prediction accuracy.
- Provide insights into factors affecting TED Talk popularity.

## Datasets

The project uses data from the [TED Talks Kaggle Dataset](https://www.kaggle.com/rounakbanik/ted-talks), which includes:
- Talk transcripts
- Metadata (e.g., speaker occupation, tags, duration, views)

## Methods

1. **Text Preprocessing**:
   - Tokenization and lemmatization using SpaCy.
   - Removal of stop words and non-alphabetic tokens.
   - TF-IDF vectorization with n-grams.

2. **Machine Learning Models**:
   - Logistic Regression (L1, L2, ElasticNet penalties).
   - Gradient Boosting Classifier.
   - Clustering models (KMeans++).
   - SpaCy-based text classification.

3. **Clustering and Dimensionality Reduction**:
   - KMeans clustering with optimal k determined using the Elbow Method.
   - PCA for visualization and dimensionality reduction.

## Results

- **Best Model**: Gradient Boosting Classifier with an accuracy of 70%.
- Logistic Regression achieved ~60% accuracy but showed convergence issues.
- SpaCy models struggled with accuracy between 50-52%, indicating limitations for this dataset.
- KMeans clustering achieved a silhouette score of 0.39, showing moderate clustering quality.

## Key Visualizations

- Distribution plots for views, comments, and durations.
- WordClouds for top themes and tags.
- Scatter plots to explore relationships between views, comments, and durations.
- Confusion matrices for evaluating model performance.

## Tools and Libraries

- **NLP**: SpaCy, NLTK
- **Machine Learning**: Scikit-learn, imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Preprocessing**: TfidfVectorizer, PCA
- **Dataset Handling**: Pandas, NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ted-talk-popularity-nlp.git
   cd ted-talk-popularity-nlp
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the working directory.

4. Run the notebook for training, evaluation, and visualization.

## Future Improvements

- Explore more advanced NLP models like Transformers (e.g., BERT).
- Implement additional metadata features for better context understanding.
- Experiment with real-time TED Talk popularity prediction tools.

## Author

**Dineth Hettiarachchi**

## License

This project is licensed under the MIT License.
