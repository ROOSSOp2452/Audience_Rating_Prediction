# Audience Rating Prediction

This project aims to predict the audience rating of movies based on various features such as genre, previous audience ratings, and other relevant factors. The model is built using machine learning algorithms and is evaluated for its performance using common metrics such as accuracy and confusion matrix.

## Project Overview

The purpose of this project is to create a predictive model that can forecast audience ratings based on historical data. This is done using machine learning algorithms like Decision Trees, Random Forests, and Gradient Boosting.

## Requirements

To run this project, you will need to have the following libraries installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Files

- **data.csv**: Dataset containing movie-related information and audience ratings.
- **model.py**: The Python script that loads the dataset, trains the model, and evaluates its performance.
- **predictions.py**: The script that generates predictions for new data based on the trained model.
- **visualization.py**: The script that plots various graphs to analyze model performance.

## How to Run

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/audience-rating-prediction.git
cd audience-rating-prediction
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Load the data and train the model:

```bash
python model.py
```

4. Generate predictions on the test set:

```bash
python predictions.py
```

5. Visualize the results:

```bash
python visualization.py
```

## Model

In this project, a Decision Tree model is used to predict audience ratings. The model is trained on historical data and is evaluated using a test set. The evaluation metrics include:

- Accuracy score
- Confusion matrix
- Classification report

## Evaluation

The performance of the model is evaluated on various metrics such as accuracy, precision, recall, and F1-score. These metrics give an understanding of how well the model is predicting the audience ratings.

## Future Improvements

- Implementing other machine learning algorithms like Random Forests and Gradient Boosting for better performance.
- Hyperparameter tuning to optimize the model's accuracy.
- Collecting more data to improve the model’s generalization capabilities.

