# Legal Document Classifier - NLP Code Challenge

## Overview
This project implements a Natural Language Processing (NLP) pipeline to classify legal case reports into areas of law, as part of the AI Engineer Role Suitability Test for LawPavilion. The pipeline processes the provided dataset (`sample_200_rows.csv`), performs preprocessing, trains multiple classification models, evaluates performance, and provides a FastAPI endpoint for inference. The solution is designed to meet the evaluation criteria: Data Preprocessing (20%), Model Performance (30%), Code Quality & Structure (20%), Clarity of Comments/README (10%), and Bonus API (20%).

## Project Structure
```
legal-document-classifier/
.
├── app
│   ├── __pycache__
│   │   └── api.cpython-310.pyc
│   └── api.py
├── data
│   └── sample_200_rows.csv
├── docs
│   └── NLP_Code_Challenge.docx.pdf
├── LICENSE
├── models
│   ├── bert_model
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── bert_tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── best_bert_label_map.json
│   ├── best_bert_model
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── best_bert_tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   └── vocab.txt
│   ├── best_label_map.json
│   ├── lr_vectorizer.pkl
│   ├── saved_lr_model.pkl
│   ├── saved_model.pkl
│   └── vectorizer.pkl
├── notebooks
│   └── code.ipynb
├── README.md
├── reports
│   ├── non-technical.md
│   └── tecnical.md
├── requirements.txt
├── streamlit_app
│   └── app.py
└── visuals
    ├── accuracy_comparison.png
    ├── confusion_matrices.png
    └── label_distribution.png
```

## Dataset
The dataset (`sample_200_rows.csv`) contains legal case reports with the following columns:
- `case_title`: Title of the case
- `suitno`: Case reference number
- `introduction`: Brief introduction (contains area of law)
- `facts`: Summary of case facts
- `issues`: Legal issues considered
- `decision`: Judgment outcome
- `full_report`: Full text of the judgment

**Input**: `full_report` is used for classification.
**Label**: Area of law extracted from `introduction` (e.g., Civil Procedure, Enforcement of Fundamental Rights).

## Approach
1. **Preprocessing**:
   - Cleaned `full_report` by removing legal citations, special characters, and normalizing text.
   - Standardized labels from `introduction` into five categories: Civil Procedure, Enforcement of Fundamental Rights, Election Petition, Garnishee Proceedings, Other.
   - Conducted EDA to analyze label distribution and text characteristics.
2. **Modeling**:
   - **Baseline**: Used TF-IDF vectorization with Logistic Regression, SVM, and Random Forest models.
   - **Advanced**: Fine-tuned a BERT model for improved performance on nuanced legal texts.
   - Evaluated models using Accuracy, F1-Score, and Confusion Matrix.
3. **Inference**:
   - Saved the Logistic Regression model and TF-IDF vectorizer for API use.
   - Implemented a FastAPI endpoint in `app/api.py` for predicting the area of law from new case reports.
4. **Evaluation**:
   - Detailed metrics provided for all models, with visualizations saved (e.g., `label_distribution.png`).
   - Logistic Regression selected for API due to its balance of performance and efficiency.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd legal-document-classifier
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   - Open `notebooks/code.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure `sample_200_rows.csv` is in the root directory.
   - Execute all cells to preprocess data, train models, and save the model/vectorizer.
4. **Run the API**:
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```
   - Access the API at `http://localhost:8000`.
   - Test the `/predict` endpoint using a tool like `curl` or Postman:
     ```bash
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"full_report": "Your legal case text here"}'
     ```
   - Check API health at `http://localhost:8000/health`.
5. **Model Files**:
   - The trained model (`saved_model.pkl`) and vectorizer (`vectorizer.pkl`) are saved in the `models/` directory.
   - The BERT model (`bert_model/`) and tokenizer (`bert_tokenizer/`) are saved for reference but not used in the API.

## Usage
- **Notebook**: Run `code.ipynb` to preprocess the dataset, train models, and evaluate performance. The notebook includes detailed comments and visualizations.
- **API**: Use the `/predict` endpoint to classify new case reports. Example request:
  ```json
  {
    "full_report": "This is a sample legal case report text..."
  }
  ```
  Response:
  ```json
  {
    "area_of_law": "Civil Procedure",
    "confidence": 0.92
  }
  ```
- **Evaluation**: Metrics (Accuracy, F1-Score, Confusion Matrix) are printed in the notebook. Visualizations are saved for reference.

## Results
- **Preprocessing**: Handled legal-specific noise (e.g., citations) and standardized labels for consistency.
- **Modeling**:
  - **TF-IDF Models**: Logistic Regression achieved robust performance, with SVM and Random Forest as alternatives.
  - **BERT**: Improved accuracy for complex legal texts but requires more computational resources.
- **API**: FastAPI endpoint is efficient, with error handling and confidence scores.
- **Code Quality**: Modular code with extensive comments and a clear structure.
- **Documentation**: This README and notebook comments ensure clarity and reproducibility.

## Future Improvements
- Use domain-specific models like Legal-BERT for better performance.
- Implement ensemble methods to combine TF-IDF and BERT predictions.
- Optimize API for scalability using Docker or cloud deployment.
- Address potential label imbalance with techniques like SMOTE.

## Contact
For questions or issues, contact [your-email@example.com] or submit an issue on the GitHub repository.

---
Submitted by: Segun Oni
For: LawPavilion AI Engineer Role Suitability Test  
Deadline: May 28, 2025