# Application Concept

The application is designed to assist educational institutions in predicting student outcomes based on various features such as demographic information, academic performance, and socio-economic factors. By using machine learning techniques, the application provides predictions on whether a student is likely to graduate, drop out, or remain enrolled.

## Key Features

1. Prediction for Candidates: Helps prospective students understand their chances of success in a particular course.
2. Prediction for Current Students: Assists universities in identifying students at risk of dropping out and taking proactive measures.
3. Course Recommendations: Suggests alternative courses for students predicted to drop out, potentially leading to better outcomes.
4. Scholarship Eligibility: Evaluates the likelihood of a student qualifying for a scholarship based on their predicted success.

## Project Structure

- `ada_boost_model.pkl`: AdaBoost model file.
- `app.py`: Main application script.
- `bagging_model_without_curriculum.pkl`: Bagging model file without curriculum.
- `bagging_model.pkl`: Bagging model file.
- `country_to_code.json`: JSON file mapping country names to codes.
- `data.csv`: Dataset used for training and evaluation.
- `decisionTree.ipynb`: Jupyter notebook for decision tree analysis.
- `decisionTrees.ipynb`: Jupyter notebook for decision trees analysis.
- `using_decision_trees.ipynb`: Jupyter notebook demonstrating the use of decision trees.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages, can be installed using `requirements.txt`, like this: 
 ```sh
pip install -r requirements.txt
```
### Running the Application


```sh
streamlit run app.py
```

### Notebooks

You can explore the Jupyter notebooks for detailed analysis and model usage:

- `decisionTrees.ipynb`
- `using_decision_trees.ipynb`

## Models

- `ada_boost_model.pkl`: Pre-trained AdaBoost model.
- `bagging_model_without_curriculum.pkl`: Pre-trained Bagging model without curriculum.
- `bagging_model.pkl`: Pre-trained Bagging model.

## Data

- `country_to_code.json`: Contains country to code mappings.
- `data.csv`: The dataset
