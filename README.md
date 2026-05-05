# Cardiovascular Disease Early Detection
Binary classification of cardiovascular disease using logistic regression,
Random Forest, and a neural network on the Kaggle Cardiovascular Disease
dataset (70,000 patients, 10 features).

## Pipeline
1. **Data preprocessing** - outlier removal (~2% of rows); `id` dropped; BMI introduced as a derived feature replacing height and weight; classes are near-perfectly balanced
2. **Feature selection** - information gain + Pearson/Spearman correlation; `smoke`, `alco`, and `gender` excluded as least informative
3. **Preprocessing** - continuous features standardized; ordinal categorical features one-hot encoded
4. **Classification** - all three models trained with cost-sensitive learning (`class_weight 1:2`) and decision threshold tuning on the validation set to keep FNR ≤ 15%

## Results

| Metric      | Logistic regression | Random Forest | Neural network |
|-------------|:-------------------:|:-------------:|:--------------:|
| Accuracy    | 67.92%              | 70.14%        | 69.53%         |
| Sensitivity | 87.75%              | 85.18%        | 85.07%         |
| Specificity | 48.50%              | 55.41%        | 54.30%         |
| FNR         | 12.25%              | 14.82%        | 14.93%         |
| AUC         | 0.793               | 0.801         | 0.800          |

Random Forest achieved the best overall balance of AUC, accuracy, and false-negative
rate, and is the recommended model for this dataset.

The main challenge was achieving acceptable accuracy while keeping FNR low. The trade-off is an increase in
false positives, which means the model is best suited as a **screening tool** - flagging
high-risk patients for further clinical evaluation rather than making definitive
diagnoses. Future improvements could include additional clinically-informed features
and a real-time prediction interface for medical practitioners.

## Files
- `praksa.ipynb` - full implementation notebook
- `praksa.pdf` - detailed project report (Serbian)

## Requirements
numpy, pandas, matplotlib, seaborn, scikit-learn, keras
