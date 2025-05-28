### Technical Report

As part of my submission for the AI Engineer role at LawPavilion, I developed a Legal Document Classifier to categorize legal case reports into areas of law using a comprehensive NLP pipeline. My approach involved preprocessing, modeling with multiple algorithms, evaluation, and the creation of an API for inference, all detailed in the provided Jupyter notebook (`code.ipynb`) and API files (`api.py` and `app.py`).

#### **Methodology**
I began by loading and exploring the dataset (`sample_200_rows.csv`), which contains 200 rows with columns like `case_title`, `introduction`, and `full_report`. I noticed significant class imbalance and varying text quality, so I implemented a text-cleaning function to remove HTML entities, citations, and special characters, standardizing the `full_report` column. To extract labels, I designed a keyword-based extraction method from the `introduction` column, mapping terms like "civil procedure" or "criminal law" to standardized categories, with a fallback to "Other" for unclassified cases.

For modeling, I experimented with traditional machine learning and deep learning approaches. I used TF-IDF vectorization with Logistic Regression, SVM, and Random Forest, applying SMOTE to address imbalance. Additionally, I fine-tuned a BERT model, optimizing it with class weights and early stopping. To enhance performance, I incorporated data augmentation with synonym replacement and hyperparameter tuning via GridSearchCV. I evaluated models using accuracy, F1-score, and confusion matrices, generating visualizations to assess results.

#### **Implementation**
I built an API using FastAPI (`api.py`) with endpoints for single and batch predictions, supporting both TF-IDF and BERT models. The API includes input validation, error handling, and logging for debugging. For user interaction, I created a Streamlit app (`app.py`) that provides a chat-like interface, allowing users to input text, select models, and view predictions with confidence scores. I saved the best models (Logistic Regression from Training 1 and BERT from Training 1) with their respective vectorizers and tokenizers for deployment.

#### **Results**
The best traditional model, **Logistic Regression** from Training 1, achieved an accuracy of 0.7000 and an F1-score of 0.6978, outperforming **BERT**, which reached 0.5000 accuracy and 0.3330 F1-score. This gap highlights the challenge of fine-tuning BERT with a small dataset (200 samples). Confusion matrices and accuracy comparisons were saved as PNG files for analysis. Memory management was addressed with garbage collection, keeping usage below 1GB.

#### **Challenges and Solutions**
I faced issues with BERT’s underperformance due to limited data and hardware constraints (CPU-only on a MacBook Pro 2020). I mitigated this by reducing batch sizes, epochs, and max_length, and switching to RoBERTa with focal loss. However, results remained suboptimal, suggesting a need for more data or a pre-trained legal-specific model like Legal-BERT.

#### **Future Work**
I plan to expand the dataset, explore Legal-BERT, and optimize the API with Docker for scalability. I also intend to add real-time usage tracking to the `/stats` endpoint.

