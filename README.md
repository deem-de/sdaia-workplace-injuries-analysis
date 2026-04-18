# Machine Learning-Based Analysis of Workplace Injuries in Saudi Arabia

> **Department of Computer Science — Princess Nourah Bint Abdulrahman University**

---

## Project Overview

This project applies machine learning techniques to analyze workplace injury data in Saudi Arabia (Q2 2024), aiming to uncover hidden patterns and identify high-risk occupational groups. By combining statistical analysis with data-driven pattern discovery, the study supports the development of targeted occupational safety strategies.

---

## Project Files

| File | Description |
|------|-------------|
| `Machine_learning_project.ipynb` | Main Jupyter notebook containing all code |
| `InjuresQ22024 xlsx 3(Sheet1).csv` | Dataset from SDAIA Open Data Portal (Q2 2024) |
| `ML_Paper_Team_NMS.pdf` | Research paper documenting the study |

---

## Dataset

- **Source:** Saudi Data and AI Authority (SDAIA) — Open Data Portal, Q2 2024
- **Size:** 5,000+ records, 13 attributes

| Column | Type | Description |
|--------|------|-------------|
| `Sex` | Categorical | Worker gender |
| `Age Group` | Categorical | Worker age range |
| `Career Category` | Categorical | Job category / occupation |
| `Office` | Categorical | Region / office location |
| `Infection Classification` | Categorical | Injury classification type |
| `Cause of Injury` | Categorical | Root cause of the injury *(main target)* |
| `Treatment Result` | Categorical | Medical outcome of the injury |
| `Number of Work Injuries` | Numeric | Count of injuries per record |

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend imbalanced-learn
```

Place the CSV file in the same directory as the notebook, or update the path:

```python
df = pd.read_csv("InjuresQ22024 xlsx 3(Sheet1).csv")
```

---

## How to Run

Run the notebook cells in the following order:

| Step | Section | Description |
|------|---------|-------------|
| 1 | Loading & Preprocessing | Load data, drop irrelevant columns, fix Age Group errors, apply LabelEncoder |
| 2 | EDA — Pie & Bar Charts | Visualize distributions of all categorical features |
| 3 | Heatmap | Explore relationship between Career Category and Cause of Injury |
| 4 | PCA Feature Ranking | Identify highest-variance features using Principal Component Analysis |
| 5 | Random Forest Importance | Rank features by predictive power toward Cause of Injury |
| 6 | K-Means Clustering | Discover hidden groups (k=4) based on top features |
| 7 | Association Rules (Apriori) | Mine Career → Cause rules with support & confidence metrics |
| 8 | SVM Classifier | Classify injury cause using SVM with RBF kernel + SMOTE balancing |

---

## Key Findings

| Method | Findings |
|--------|----------|
| **PCA** | Cause of Injury (0.676) and Infection Classification (0.666) are highest-variance features |
| **Random Forest** | Injury Type (0.40) and Office/Region (0.25) are most predictive |
| **K-Means (k=4)** | 4 clusters map to distinct injury patterns by occupation type |
| **Association Rules** | Career categories consistently link to specific injury causes |
| **Demographics** | Non-Saudi male workers in construction & industrial sectors show highest injury rates |

---

## Important Notes

- **Age Group fix:** The value `#CONNECT!` is automatically replaced with `29 ~ 25` during preprocessing.
- **SVM:** Run the SMOTE cell before SVM training to handle class imbalance.
- **Data leakage:** SMOTE is applied only on training data (after train-test split).
- **Paper gap:** Section IV (Recommendations) in the research paper is currently empty.

---

## Authors

| Name | Email |
|------|-------|
| Jumana Assweed | 445008859@pnu.edu.sa |
| Dana Alsubaie | 445008870@pnu.edu.sa |
| Shomokh Alotaibi | 445008903@pnu.edu.sa |
| Deem Alrashoud | 445008860@pnu.edu.sa |
