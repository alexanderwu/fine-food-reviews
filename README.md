# Rating Prediction of Fine Food Reviews

In this project, I predict rating score using various models:

* Sentiment analysis models (like Unigram-Bigram word count and TF-IDF model)
* Latent factor models
* Linear regression

Evaluating each of the above models using Mean Squared Error (MSE) on a test set, I found Unigram-Bigram TF-IDF model to perform the best with MSE of 0.7217.

This was a group project from "CSE 258 - Recommender Systems" class I took at UC San Diego in Fall quarter 2018. I worked on the making the graphs for exploratory data analysis (using matplotlib and seaborn) as well as the sentiment analysis models (using sklearn).

## Overview

Here are the files in this project:

* **FineFoodReviews.ipynb**: Jupyter notebook for project
* **sklearn_text_models.py**: Python script to train Unigram-Bigram TF-IDF model and report test MSE
* **FinalReport.pdf**: Written report

## Getting Started

To run the Jupyter Notebook, you will need to retrieve `Reviews.csv` from the [Amazon fine foods reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews) hosted by Kaggle. The data contains 568,454 rows and 10 columns.

For convenience, save the data frame into a pickle file to quickly load the dataset next time. For example, the following command will save a file called 'data.pkl' into the current working directory.

```py
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

Then, you may read in the pickle file into a variable `data` like so:

```py
with open('data.pkl','rb') as file:
    data = pickle.load(file)
```

These steps will help for the `sklearn_text_models.py` script
