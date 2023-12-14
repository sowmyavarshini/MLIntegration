# Purpose
This script demonstrates the use of a Decision Tree Classifier to classify iris flowers based on their features. The primary objectives are:

- Load the Iris dataset.
- Split the dataset into training and testing sets.
- Train a Decision Tree Classifier on the training set.
- Make predictions on the test set.
- Evaluate the model's performance using accuracy and a classification report.

# Chosen Model

The script utilizes the DecisionTreeClassifier from scikit-learn. Decision trees are chosen for their simplicity, interpretability, and effectiveness in classification tasks.

# Chosen Dataset

The dataset contains: 3 classes (different Iris species) with 50 samples each, and then four numeric properties about those classes: Sepal Length, Sepal Width, Petal Length, and Petal Width of flowers: setosa, versicolor, and virginica.

# Steps

## Loading the dataset
The script loads the Iris dataset using the load_iris function from scikit-learn. Features (X) and target labels (y) are extracted.

## Splitting the Dataset
The dataset is split into training and testing sets using the train_test_split function. 80% of the data is used for training, and 20% is reserved for testing.

## Training the Model
A Decision Tree Classifier is trained on the training set using the DecisionTreeClassifier class.

```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```
## Making Predictions

```python
y_pred = model.predict(X_test)
```
## Evaluating Perfomance
```python
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
```
# Parameters and Configurations

- The test size for splitting the dataset is set to 20%.
- A random state of 42 is used for reproducibility in the train-test split.

# How to run 
- Ensure you have the parameters installed
  ```bash
  pip install scikit-learn
  ```
-  Run the script
```python
  python main.py
```
