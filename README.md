# PriceBreakthroughsAI

## Overview

**PriceBreakthroughsAI** is a machine learning project focused on predicting price breakthroughs in financial markets using a combination of decision tree-based models (such as XGBoost and Decision Trees), random forest models, and transformer-based models. The goal is to accurately classify whether a significant price movement will occur within a specified timeframe, which is essential for developing robust trading strategies.

## Motivation

The financial markets are highly dynamic and often exhibit sudden and significant price movements. Predicting these movements, known as price breakthroughs, can be highly beneficial for traders and financial analysts. By leveraging advanced machine learning techniques, this project aims to create models that can predict these price breakthroughs with a high degree of accuracy, thus providing a valuable tool for decision-making in trading and investment.

## Data

The data used for this project is sourced from [Kaggle's BTC/USD Historical Data](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd). This dataset includes historical price data for Bitcoin (BTC) against the US Dollar (USD), which has been preprocessed to create the features used in the models.


## Results

### Decision Tree Model
- **Accuracy:** 93.4%
- **Confusion Matrix:**

|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| **Actual 0**| 139,800     | 3,726       |
| **Actual 1**| 6,133       | 280         |

- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.96      | 0.97   | 0.97     | 143,526 |
| 1     | 0.07      | 0.04   | 0.05     | 6,413   |

### XGBoost Model
- **Accuracy:** 82.0%
- **Confusion Matrix:**

|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| **Actual 0**| 120,680     | 22,846      |
| **Actual 1**| 4,112       | 2,301       |

- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.97      | 0.84   | 0.90     | 143,526 |
| 1     | 0.09      | 0.36   | 0.15     | 6,413   |

### Random Forest Model
- **Accuracy:** 95.7%
- **Confusion Matrix:**

|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| **Actual 0**| 143,526     | 0           |
| **Actual 1**| 6,413       | 0           |

- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.96      | 1.00   | 0.98     | 143,526 |
| 1     | 0.00      | 0.00   | 0.00     | 6,413   |

### Transformer-Based Model
- **Note:** Transformer-based models can achieve results similar to those of decision tree, XGBoost, and random forest models, depending on the training configuration. These models are particularly useful for capturing temporal patterns in time series data, which can enhance breakthrough prediction accuracy.

## Analysis and Considerations
The results highlight significant differences in performance among the models:

- **Decision Tree Model:**  
  The decision tree achieved a high overall accuracy of 93.4%, with a strong ability to predict class 0 (no breakthrough). However, its performance in predicting class 1 (breakthrough) was weaker, with a low F1-score of 0.05.

- **XGBoost Model:**  
The XGBoost model had an accuracy of 82.0%, which is lower than that of the decision tree. However, it showed improved recall and F1-score for class 1, achieving a recall of 0.36 and an F1-score of 0.15 compared to the decision tree's 0.05, respectively. This indicates that XGBoost is more effective at identifying breakthroughs, making it a better choice when predicting class 1 is the priority.

- **Random Forest Model:**  
  The random forest model showed a high accuracy of 95.7%, but it struggled to predict class 1 (breakthrough), resulting in an F1-score of 0.00 for this class. This indicates that while the model fails to identify actual breakthroughs.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PriceBreakthroughsAI.git

python scripts/train_tree.py
python scripts/train_transformer.py



## Model Evaluation and Comparison

The models were evaluated based on accuracy, precision, recall, and F1-score. Below is a comparison of the models:

| Model           | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-----------------|----------|---------------------|------------------|--------------------|
| Decision Tree   | 93.4%    | 0.07                | 0.04             | 0.05               |
| XGBoost         | 82.0%    | 0.09                | 0.36             | 0.15               |
| Random Forest   | 95.7%    | 0.00                | 0.00             | 0.00               |

### 4. **Future Work**
   - Outline potential future improvements or extensions of the project.

## Future Work

- **Hyperparameter Tuning:** Further fine-tuning of model hyperparameters to improve performance, particularly for class 1 predictions.
- **Model Ensemble:** Combining the strengths of multiple models (e.g., using an ensemble of XGBoost and Decision Trees) to enhance prediction accuracy.
- **Real-time Predictions:** Implementing real-time predictions to be used in live trading environments.
- **Extended Data:** Incorporating more data, including other cryptocurrencies or additional features, to improve model generalization.

### 5. **Contributing**
   - If this is an open-source project, include a section on how others can contribute.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. You can also open an issue if you find a bug or have a question.

### 6. **License**
   - Reiterate the license under which the project is released.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 7. **Acknowledgments**
   - Give credit to any libraries, tools, or individuals that contributed to the project.

## Acknowledgments

- **Scikit-learn** and **XGBoost** for providing the machine learning algorithms used in this project.
- **Kaggle** for providing the dataset.
- **Matplotlib** and **Seaborn** for visualization tools.

## Future Work

- **Hyperparameter Tuning:** Further fine-tuning of model hyperparameters to improve performance, particularly for class 1 predictions.
- **Model Ensemble:** Combining the strengths of multiple models (e.g., using an ensemble of XGBoost and Decision Trees) to enhance prediction accuracy.
- **Real-time Predictions:** Implementing real-time predictions to be used in live trading environments.
- **Extended Data:** Incorporating more data, including other cryptocurrencies or additional features, to improve model generalization.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. You can also open an issue if you find a bug or have a question.

For cooperation regarding the project, please contact aleksandrsimonyan1996@gmail.com or connect with me on [LinkedIn](https://www.linkedin.com/in/alekssim/).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Scikit-learn** and **XGBoost** for providing the machine learning algorithms used in this project.
- **Kaggle** for providing the dataset.
- **Matplotlib** and **Seaborn** for visualization tools.
