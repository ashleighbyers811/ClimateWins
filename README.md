# ClimateWins: Machine Learning Analysis

## Project Summary
This repository contains an analysis aimed at predicting climate change outcomes using machine learning models trained on historical weather data. The analysis focuses on identifying climate patterns that can inform planning and adaptation strategies, particularly for extreme weather events and renewable energy opportunities.

## Objectives
1. **Identifying Regional Anamalies**: Detecting weather patterns that fall outside normal regional trends to understand emerging risks.
2. **Assessing the Increase in Unusual Weather Events**: Analyzing trends to determine if extreme weather occurrences are becoming more frequent or severe.
3. **Forecasting Future Climate Scenarios**: Projecting future weather conditions over the next25 to 50 years based on current climate trends.
4. **Identifying Safe Havens**: Determining the safest places for people to live by understanding which areas will be most resilient to climate change in the coming decades.

## Key Questions
1. **Extreme Weather Prediction**: Can machine learning models identify patterns in wind speed, pressure, and precipitation that signal the potential formation of typhoons or hurricanes in vulnerable regions?
2. **Solar Energy Forecasting**: Can models trained on global radiation and cloud cover data forecast high-sunshine days, which may indicate increased solar energy potential?
3. **Seasonal Pattern Shifts**: Can long-term trends in temperature, precipitation, and snow depth help predict changes in seasonal patterns, such as longer summers or shorter winters, and their impact on ecosystems?

## Folders
The repository is structured as follows:

- **01 Project Management**: Contains project brief.
- **02 Data**: Contains raw and prepped data.
- **03 Scripts**: Python scripts for data cleaning, feature engineering, and model training. Contains subfolders:
  - **Supervised**: Scripts for supervised machine learning models.
  - **Unsupervised**: Scripts for any unsupervised learning approaches.
- **05 Sent To Client**: Contains all reports and presentations for stakeholders.

## Code Overview
The code for this project is written in Python and executed in Jupyter notebooks, utilizing a range of libraries for data manipulation, visualization, machine learning, and evaluation. Key libraries and their purposes are as follows:

- **Pandas**: For data manipulation, cleaning, and transformation.
- **NumPy**: For efficient numerical calculations and array handling.
- **OS**: For interaction with the operating system, including file and directory operations.
- **Seaborn and Matplotlib**: For data visualization, including statistical plots and custom figure management.
- **Scikit-learn**: For machine learning model training, evaluation, and metrics, specifically:
  - **KNeighborsClassifier**: Implements K-Nearest Neighbors for classification tasks.
  - **DecisionTreeClassifier**: Utilizes decision tree algorithms to classify data based on feature splits.
  - **MultiOutputClassifier**: Supports multi-output classification tasks, applying classifiers to each target variable.
  - **MLPClassifier**: Applies a multilayer perceptron for neural network-based classification.
  - **StandardScaler**: Standardizes features to have zero mean and unit variance, improving model performance.
  - **Metrics and Model Selection**: Including `train_test_split`, `cross_val_score`, `multilabel_confusion_matrix`, `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, and `classification_report` to evaluate model accuracy, handle multi-label classification, and visualize confusion matrices.
- **Graphviz**: For visualizing decision tree structures, enhancing interpretability of classification rules.

These tools support data exploration, model training, and performance assessment, enabling comprehensive analysis and insights into climate change patterns.

## Models Evaluated
The project evaluates a range of supervised machine learning models, including:
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Artificial Neural Network (ANN)**
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**
- **Random Forest**

These models are compared and assessed based on accuracy, interpretability, and suitability for climate outcome prediction.
