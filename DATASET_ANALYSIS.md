# Forward Osmosis Membrane Flux Dataset Analysis

## 📊 Dataset Overview

This dataset contains experimental data for **Forward Osmosis (FO) membrane flux prediction**. The data is stored in an Excel file with multiple sheets containing different views and processing stages of the experimental results.

---

## 📁 File Structure

**File:** `Supplementary Data.xlsx`

### Sheets:
1. **Raw_data** - Main dataset with 856 experimental observations
2. **Input_data** - Processed/encoded data (798 rows, 45 columns) - appears to be one-hot encoded
3. **Comparison** - Model comparison data (30 rows)
4. **Bibliography** - Empty sheet (likely for references)

---

## 🔬 Main Dataset: Raw_data

### Dimensions
- **Total Rows:** 856 experimental observations
- **Total Columns:** 21 features

### Column Description

#### 1. **Metadata (2 columns)**
| Column | Type | Description |
|--------|------|-------------|
| Author | Text | Research author/paper reference |
| Year | Numeric | Publication year (2006-2012) |

#### 2. **Membrane Characteristics (3 columns)**
| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| Membrane AL Type | Categorical | Active layer type | CTA, TFC |
| Manufacture | Categorical | Membrane manufacturer | HTI, etc. |
| Membrane Direction | Categorical | Membrane orientation | AL-FS, AL-DS |

#### 3. **Experimental Setup (2 columns)**
| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| Flow Orientation | Categorical | Flow pattern | cocurrent, counter |
| Feed Solution (FS) | Categorical | Feed solution type | DI, NaCl, etc. |

#### 4. **Draw Solution Properties (4 columns)**
| Column | Type | Description | Unit |
|--------|------|-------------|------|
| Draw Solution (DS) | Categorical | Draw solution type | - |
| DS Molarity (M) | Numeric | Concentration | M (Molar) |
| DS Molecular weight | Numeric | Molecular weight | g/mol |
| DS Osmotic pressure (atm) | Numeric | Osmotic pressure | atm |

#### 5. **Feed Solution Properties (2 columns)**
| Column | Type | Description | Unit |
|--------|------|-------------|------|
| FS Molarity (M) | Numeric | Concentration | M (Molar) |
| FS Osmotic pressure (atm) | Numeric | Osmotic pressure | atm |

#### 6. **Operating Conditions (5 columns)**
| Column | Type | Description | Unit |
|--------|------|-------------|------|
| P Difference (atm) | Numeric | Pressure difference | atm |
| FS Velocity (cm/s) | Numeric | Feed solution velocity | cm/s |
| DS Velocity (cm/s) | Numeric | Draw solution velocity | cm/s |
| FS Temperature (°C) | Numeric | Feed solution temperature | °C |
| DS Temperature (°C) | Numeric | Draw solution temperature | °C |

#### 7. **Membrane Parameters & Target (3 columns)**
| Column | Type | Description | Unit |
|--------|------|-------------|------|
| A (LMH/atm) | Numeric | Water permeability coefficient | LMH/atm |
| S (um) | Numeric | Structural parameter | μm |
| **Membrane Flux (LMH)** | **Numeric** | **Target variable - Water flux** | **LMH** |

> **LMH** = Liters per square Meter per Hour

---

## 📈 Data Statistics

### Sample Data (First 10 rows)
All experiments from **Gray (2006)** using:
- **Membrane:** CTA from HTI
- **Orientation:** AL-FS (Active Layer facing Feed Solution)
- **Flow:** Cocurrent
- **Feed:** Deionized water (DI)
- **Draw:** Dextrose or Sucrose solutions
- **Temperature:** 22.5°C (both sides)
- **Velocities:** 30 cm/s (both sides)

### Value Ranges
- **Years:** 2006-2012
- **DS Molarity:** 0.11 - 1.78 M (in sample)
- **DS Osmotic Pressure:** 2.72 - 43.58 atm (in sample)
- **Membrane Flux:** 2.67 - 43.15 LMH (in sample)

---

## ⚠️ Missing Values Analysis

| Column | Missing Count | Percentage | Severity |
|--------|---------------|------------|----------|
| **S (um)** | 730 | **85.3%** | 🔴 Critical |
| **A (LMH/atm)** | 148 | **17.3%** | 🟡 Moderate |
| **FS Velocity (cm/s)** | 28 | **3.3%** | 🟢 Minor |
| **DS Velocity (cm/s)** | 28 | **3.3%** | 🟢 Minor |

### Key Observations:
- ✅ **Target variable** (`Membrane Flux (LMH)`) has **NO missing values**
- 🔴 **Structural parameter (S)** is mostly missing - may need imputation or exclusion
- 🟡 **Water permeability (A)** has moderate missingness
- 🟢 **Velocity parameters** have minimal missingness

---

## 🎯 Dataset Purpose

This dataset appears to be designed for:

1. **Predictive Modeling** - Predicting membrane flux based on operating conditions
2. **Process Optimization** - Understanding how different parameters affect flux
3. **Membrane Comparison** - Comparing performance across different membrane types and manufacturers

---

## 🔍 Additional Sheets

### Input_data Sheet
- **798 rows × 45 columns**
- Appears to be **one-hot encoded** version of Raw_data
- Columns are numbered (0-36) plus some named features
- Likely prepared for machine learning models

### Comparison Sheet
- **30 rows** of comparison data
- Contains both "Experiment Flux" and "SD model flux"
- Used for model validation/comparison

---

## 💡 Recommended Next Steps

### 1. **Data Cleaning & Preprocessing**
```python
# Handle missing values
- Impute or drop S (um) column (85% missing)
- Impute A (LMH/atm) using domain knowledge or statistical methods
- Impute velocity values (only 3% missing)
```

### 2. **Exploratory Data Analysis (EDA)**
- Distribution analysis of numerical features
- Correlation analysis with target variable (Membrane Flux)
- Categorical feature analysis (membrane types, manufacturers)
- Temporal trends (by year)

### 3. **Feature Engineering**
- Create interaction features (e.g., osmotic pressure difference)
- Normalize/standardize numerical features
- Encode categorical variables (already done in Input_data sheet)

### 4. **Modeling Approaches**
- **Regression models:** Linear Regression, Random Forest, XGBoost, Neural Networks
- **Target:** Membrane Flux (LMH)
- **Features:** All other columns (after preprocessing)

### 5. **Model Validation**
- Use the Comparison sheet for validation
- Cross-validation on main dataset
- Compare with existing "SD model"

---

## 📚 Domain Context

**Forward Osmosis (FO)** is a membrane separation process where:
- Water moves from **Feed Solution (FS)** to **Draw Solution (DS)**
- Driven by **osmotic pressure gradient** (not hydraulic pressure)
- Used in water treatment, desalination, food processing

**Key Factors Affecting Flux:**
1. Osmotic pressure difference
2. Membrane properties (A, S parameters)
3. Flow conditions (velocity, orientation)
4. Temperature
5. Concentration polarization effects

---

## 🛠️ Quick Start Code

```python
import pandas as pd

# Load the main dataset
df = pd.read_excel('Supplementary Data.xlsx', sheet_name='Raw_data')

# Remove unnamed columns
df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(f"Dataset shape: {df_clean.shape}")
print(f"Target variable: Membrane Flux (LMH)")
print(f"Missing values:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")
```

---

## 📊 Data Quality Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Completeness** | 🟡 Moderate | 85% missing in S parameter |
| **Consistency** | ✅ Good | Well-structured, consistent units |
| **Size** | ✅ Good | 856 observations sufficient for modeling |
| **Target Variable** | ✅ Excellent | No missing values in Membrane Flux |
| **Documentation** | ✅ Good | Clear column names and units |

---

**Generated:** 2025-11-22  
**Dataset:** Supplementary Data.xlsx  
**Total Observations:** 856  
**Target Variable:** Membrane Flux (LMH)
