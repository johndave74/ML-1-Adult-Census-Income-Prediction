# ML-1: Adult-Census-Income-Prediction
This project aims to predict whether an individual's annual income exceeds $50K based on census data using machine learning models. 

# Adult Census Income Prediction

![image](https://github.com/user-attachments/assets/57c95162-c710-40ef-b9fe-30f4bc6a08fb)

## Overview
This project aims to predict whether an individual's annual income exceeds $50K based on census data using machine learning models. The dataset is obtained from the U.S. Census Bureau and contains demographic and employment-related attributes. The classification task is performed using various machine learning algorithms, and detailed insights are provided at each step.

## Dataset Description
The dataset consists of multiple demographic and employment-related attributes. The target variable (`income`) has two categories: `<=50K` and `>50K`. Below are the key features in the dataset:

- **Age**: Age of the individual
- **Workclass**: Type of employer (e.g., private, government, self-employed)
- **Education**: Highest level of education attained
- **Marital Status**: Marital status of the individual
- **Occupation**: Job category
- **Relationship**: Relationship status (e.g., spouse, unmarried)
- **Race**: Ethnic background
- **Gender**: Male or Female
- **Hours-per-week**: Number of hours worked per week
- **Native Country**: Country of origin
- **Income**: Binary classification (`<=50K` or `>50K`)

### Dataset Source
[Kaggle: Adult Census Income Dataset](https://www.kaggle.com/datasets/lovishbansal123/adult-census-income)

---

## Libraries Used
To conduct this analysis, the following Python libraries were used:

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model training and evaluation.

---

## Step-by-Step Analysis

### Step 1: Data Loading & Understanding
- The dataset was loaded into a Pandas DataFrame.
- The shape of the dataset was checked to determine the number of rows and columns.
- Numerical and categorical columns were identified.
- Initial statistics (mean, standard deviation, missing values) were explored.

### Step 2: Exploratory Data Analysis (EDA)
EDA was performed to understand data distribution and feature importance:

- **Class Distribution**: The target variable was analyzed to check for imbalance.
- **Feature Distributions**: Histograms and boxplots were used to visualize numerical variables.
- **Correlation Analysis**: A heatmap was created to identify relationships between numerical variables.
- **Categorical Variables Analysis**: Count plots were used to examine categorical feature distributions.

### Step 3: Data Preprocessing
Data preprocessing involved cleaning and transforming the data for machine learning models:

- **Handling Missing Values**: Checked for null values and applied appropriate imputation.
- **Encoding Categorical Variables**:
  - Label encoding for binary categories.
  - One-hot encoding for multi-class categorical features (pandas `get_dummies`)

### Step 4: Model Building
The dataset was split into training and test sets (80% training, 20% testing). The following classification models were implemented:

- **Decision Tree Classifier**: Trained with `min_samples_leaf=3000` to reduce overfitting.

Hyperparameter tuning was performed to optimize model performance.

![image](https://github.com/user-attachments/assets/ff066469-935c-4b2e-aeb9-d2a5d4d9fc08)


### Step 5: Model Evaluation
The trained models were evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified instances.
- **Precision**: Measures the relevance of positive predictions.
- **Recall**: Measures how many actual positive instances were identified.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: A visualization of classification errors.

### Results & Insights
- **Model Performance**:
  - Decision Tree model achieved an accuracy of **76%**.
  - ConfusionMatrix showing the TP (True Positive), FP (False Positive), TN (True Negative), FN (False Negative)

![image](https://github.com/user-attachments/assets/2acaa9f7-bd28-4fbe-99e0-a906a95456ba)


---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adult_income_prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train and evaluate the model.

---

## Future Improvements
- **Hyperparameter tuning** for better accuracy.
- **Using Deep Learning models** such as Neural Networks.
- **Adding additional socio-economic features** for improved prediction.

## Author
[John David]

## License
This project is licensed under the MIT License.
