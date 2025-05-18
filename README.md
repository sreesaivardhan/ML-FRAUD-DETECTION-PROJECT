# ML Fraud Detection Project

This project is focused on detecting fraudulent transactions using machine learning techniques. The dataset consists of real-world transaction data, and the goal is to build a model that can accurately classify transactions as fraudulent or legitimate.

## Project Structure
- `FINAL.ipynb`: Final notebook with complete workflow and results.
- `Training.ipynb`: Notebook used for model training and experimentation.
- `test.ipynb`: Additional testing and validation notebook.
- `fraudTrain.csv`: Training dataset.
- `fraudTest.csv`: Testing dataset.
- `upload.txt`: Supplementary notes or logs.

## Approach
1. **Data Preprocessing:**
   - Handling missing values
   - Feature engineering
   - Data normalization/standardization
2. **Model Selection:**
   - Various algorithms tested (e.g., Logistic Regression, Random Forest, XGBoost)
   - Hyperparameter tuning
3. **Evaluation:**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix analysis
4. **Results:**
   - Best performing model and metrics summary

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/sreesaivardhan/ML-FRAUD-DETECTION-PROJECT.git
   ```
2. Install the required Python packages (see below).
3. Open and run the notebooks in Jupyter or VS Code.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost (optional, for advanced models)

Install dependencies with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Credits
- Data source: [Kaggle - Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## License
This project is licensed under the MIT License.
