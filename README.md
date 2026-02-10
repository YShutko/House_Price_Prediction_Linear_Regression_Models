# House Price Prediction - Kansas City Housing Dataset

A comprehensive machine learning project comparing various regression models to predict house prices in Kansas City, Washington (USA).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Models Implemented](#models-implemented)
- [Key Results](#key-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Conclusions](#conclusions)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

This project explores and compares different linear regression techniques to predict house prices based on various property features. The analysis includes:

- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Multiple regression model implementations
- Hyperparameter tuning with cross-validation
- Comprehensive model comparison and evaluation

### Objectives
1. Understand the relationship between house features and prices
2. Build and evaluate multiple regression models
3. Identify the best-performing model for price prediction
4. Provide actionable insights for real estate valuation

## Dataset Information

### Source
- **Origin:** UCI Machine Learning Repository
- **Available on:** [Kaggle - KC House Data](https://www.kaggle.com/harlfoxem/housesalesprediction)

### Dataset Characteristics
- **Samples:** 21,613 house sales
- **Original Features:** 22 columns
- **Features after preprocessing:** 16 columns
- **Target Variable:** House price (USD)
- **Time Period:** House sales from 2014-2015

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| **price** | Target | Sale price of the house (USD) |
| **bedrooms** | Numeric | Number of bedrooms |
| **bathrooms** | Numeric | Number of bathrooms (0.5 = toilet only) |
| **living_qm** | Numeric | Interior living space (m²) |
| **lot_qm** | Numeric | Land lot size (m²) |
| **floors** | Numeric | Number of floors |
| **waterfront** | Binary | Waterfront property (1) or not (0) |
| **view** | Ordinal | View quality index (0-4) |
| **condition** | Ordinal | Property condition (1-5) |
| **grade** | Ordinal | Construction quality (1-13) |
| **above_qm** | Numeric | Interior space above ground level (m²) |
| **basement_qm** | Numeric | Basement space (m²) |
| **house_age** | Numeric | Age of the house (years) |
| **living15_qm** | Numeric | Avg. living space of 15 nearest neighbors (m²) |
| **lot15_qm** | Numeric | Avg. lot size of 15 nearest neighbors (m²) |

## Models Implemented

### 1. Simple Linear Regression
- **Description:** Baseline model with standard scaling
- **R² Score:** ~0.65
- **Purpose:** Establish baseline performance

### 2. Linear Regression with Cross-Validation
- **Description:** 4-fold cross-validation for robust estimation
- **R² Score:** ~0.65
- **Purpose:** Validate baseline performance

### 3. Polynomial Regression
Multiple polynomial degrees tested:
- **Degree 2:** R² ≈ 0.71 (good performance)
- **Degree 3:** R² < 0 (severe overfitting)
- **Degree 4:** R² < 0 (severe overfitting)

### 4. Ridge Regression (L2 Regularization)
- **Without Polynomial Features:** R² ≈ 0.65, α ≈ 20
- **With Polynomial Features (degree 2):** R² ≈ 0.75, α ≈ 0.1 - Winner
  - **Winner:** Best performing model!

## Key Results

### Best Model Performance
```
Model: Ridge Regression + Polynomial Features (degree 2)
R² Score: 0.7515
Optimal Alpha: 0.1
Performance: Explains 75% of house price variance
```

### Model Comparison Summary

| Model | R² Score | Notes |
|-------|----------|-------|
| Ridge + Poly (deg=2) | 0.7515 |  Best model |
| Poly (deg=2) + CV | 0.7294 | Good without regularization |
| Polynomial (deg=2) | 0.7052 | Captures non-linearity |
| Linear + CV | 0.6528 | Robust baseline |
| Ridge Regression | 0.6537 | Regularization helps |
| Simple Linear | 0.6527 | Baseline |

### Feature Importance (Top 5)
1. **living_qm** - Interior living space
2. **grade** - Construction quality
3. **above_qm** - Above-ground space
4. **bathrooms** - Number of bathrooms
5. **view** - View quality

## Methodology

### 1. Data Preprocessing
- Removed irrelevant features (ID, date, location)
- Converted square feet to square meters (SI units)
- Created house_age feature from yr_built
- No missing values detected

### 2. Exploratory Data Analysis
- Statistical summary and distribution analysis
- Correlation analysis with heatmaps
- Scatter matrix for top features
- Identified skewed features

### 3. Model Development
- Train/test split (70/30)
- Feature scaling with StandardScaler
- Cross-validation (4-fold K-Fold)
- Pipeline implementation for consistency

### 4. Hyperparameter Tuning
- Tested 1,000 alpha values for Ridge (0.1 to 20)
- Geometric spacing for efficient search
- Cross-validation for each configuration

### 5. Model Evaluation
- Primary metric: R² score
- Visualization: predictions vs. actual plots
- Comparison across all models

## Conclusions

### Key Findings

1. **Best Model:** Ridge Regression with 2nd-degree polynomial features
   - Achieves R² = 0.75
   - Optimal balance of complexity and performance
   - Regularization prevents overfitting

2. **Important Predictors:**
   - The living area is the strongest predictor
   - Construction quality (grade) significantly impacts price
   - Location features (lat/long) were excluded, but could improve the model

3. **Model Complexity:**
   - Polynomial degree 2 is optimal
   - Higher degrees (3, 4) cause severe overfitting
   - Regularization is essential with polynomial features

### Limitations

- 25% of price variance remains unexplained
- Missing features: school quality, crime rates, amenities
- No temporal analysis (seasonality, market trends)
- Geographic features excluded (could use geographic encoding)

##  Future Improvements

### Short-term
1. **Feature Engineering:**
   - Add interaction terms (e.g., bedrooms × living_area)
   - Include neighborhood statistics
   - Temporal features (month, season)

2. **Advanced Preprocessing:**
   - Handle outliers with robust methods
   - Apply log transformation to skewed features
   - Consider PCA for dimensionality reduction

### Long-term
1. **Advanced Models:**
   - Random Forest Regression (expected R² > 0.85)
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks for complex patterns

2. **Ensemble Methods:**
   - Combine multiple models (stacking)
   - Weighted averaging of predictions

3. **Production Deployment:**
   - Create REST API for predictions
   - Build web interface
   - Add confidence intervals
   - Real-time price updates

## Business Applications

This model can be used for:
- **Automated Valuation Models (AVM)** for real estate platforms
- **Investment Analysis** - identify undervalued properties
- **Market Research** - understand price drivers
- **Pricing Strategy** for real estate agents

## Dependencies

Main libraries used:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualizations
- `scikit-learn` - Machine learning models
- `scipy` - Scientific computing

See `requirements.txt` for a complete list with versions.

## Author

**Yulia Shutko**
- Project: Supervised Machine Learning - Regression Analysis
- Course: Machine Learning Specialization

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository
- Platform: Kaggle
- Tools: Jupyter, scikit-learn, Python ecosystem


**Note:** This is an educational project demonstrating regression techniques. For production use, consider more advanced models and additional validation.
