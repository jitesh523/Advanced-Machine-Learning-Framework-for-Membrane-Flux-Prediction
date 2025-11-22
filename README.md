# Membrane Flux Prediction Project

## 🎯 Project Overview
Machine learning pipeline for predicting Forward Osmosis membrane flux based on operating conditions, membrane properties, and solution characteristics.

## 📊 Dataset
- **Source**: Supplementary Data.xlsx
- **Observations**: 856 (680 after cleaning)
- **Features**: 20 (after preprocessing)
- **Target**: Membrane Flux (LMH)

## 🗂️ Project Structure

```
permute-flux-prediction-/
├── data/                          # Data files
│   ├── Supplementary Data.xlsx    # Original dataset
│   └── clean_data.csv            # Cleaned dataset
├── src/                          # Source code
│   ├── feature_engineering.py    # Feature engineering pipeline
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Evaluation utilities
│   ├── models/                   # Model implementations
│   │   ├── baseline_models.py
│   │   └── advanced_models.py
│   ├── optimization/             # Hyperparameter tuning
│   │   ├── bayesian_opt.py
│   │   └── grid_search.py
│   ├── interpretation/           # Model interpretation
│   │   └── shap_analysis.py
│   └── visualization/            # Plotting utilities
│       └── plots.py
├── models/                       # Saved trained models
├── results/                      # Results and reports
│   ├── model_comparison.md
│   └── shap_report.md
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
│   ├── DATASET_ANALYSIS.md
│   └── NEXT_STEPS.md
├── requirements.txt              # Python dependencies
├── explore_dataset.py            # Dataset exploration tool
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore Dataset
```bash
python explore_dataset.py
```

### 3. Run Full Pipeline (Coming Soon)
```bash
python src/train.py --model all --optimize bayesian
```

## 📈 Project Phases

### ✅ Phase 1: Data Preparation (Complete)
- [x] Load and explore dataset
- [x] Handle missing values (85% in S parameter)
- [x] Create clean dataset (680 observations)
- [x] Initial analysis and documentation

### 🔄 Phase 2: Feature Engineering (In Progress)
- [ ] Create derived features (osmotic pressure diff, temp diff, velocity ratio)
- [ ] One-hot encode categorical variables
- [ ] Scale/normalize numerical features
- [ ] Feature selection and correlation analysis

### 📊 Phase 3: Model Development
- [ ] Train-test split (80-20)
- [ ] Baseline models (Linear Regression, Ridge, Lasso)
- [ ] Advanced models (Random Forest, XGBoost, Neural Network)
- [ ] Model evaluation and comparison

### 🎯 Phase 4: Hyperparameter Optimization
- [ ] Grid Search for baseline models
- [ ] Bayesian Optimization for advanced models
- [ ] Cross-validation (5-fold)
- [ ] Final model selection

### 🔍 Phase 5: Model Interpretation
- [ ] SHAP analysis for feature importance
- [ ] Partial dependence plots
- [ ] Error analysis and residual plots
- [ ] Model validation on comparison data

### 📝 Phase 6: Documentation & Deployment
- [ ] Create final report with results
- [ ] Document model performance metrics
- [ ] Save trained models
- [ ] Push to GitHub

## 🎯 Target Performance (from Roadmap)

| Model | Expected R² | Status |
|-------|-------------|--------|
| Linear Regression | 0.70-0.85 | Pending |
| Random Forest | 0.90-0.95 | Pending |
| XGBoost | 0.92-0.97 | Pending |
| Neural Network | 0.94-0.99 | Pending |

## 📚 Key Features

### Input Features
- **Membrane Properties**: Type, Manufacturer, Direction, Permeability (A)
- **Operating Conditions**: Velocities, Temperatures, Pressure
- **Solution Properties**: Molarity, Osmotic Pressure, Molecular Weight

### Target Variable
- **Membrane Flux (LMH)**: Water flux through membrane

## 🛠️ Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, tensorflow
- **Optimization**: bayesian-optimization
- **Interpretability**: SHAP
- **Visualization**: matplotlib, seaborn

## 📖 Documentation

- [Dataset Analysis](DATASET_ANALYSIS.md) - Comprehensive dataset overview
- [Next Steps](NEXT_STEPS.md) - Detailed roadmap and recommendations
- [Implementation Plan](implementation_plan.md) - Technical implementation details

## 🤝 Contributing

This is a research project for membrane flux prediction. For questions or suggestions, please open an issue.

## 📄 License

This project is for educational and research purposes.

## 🔗 References

Based on research in Forward Osmosis membrane processes and machine learning applications in membrane science.

---

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄

**Last Updated**: November 22, 2025
