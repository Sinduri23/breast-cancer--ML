# Breast Cancer Relapse Prediction Using Gene Expression Data

This project focuses on predicting relapse in breast cancer patients using gene expression data and machine learning. Early identification of patients at high risk of relapse can support personalized treatment strategies and improve long-term outcomes. The model uses Random Forest classification to learn patterns from genomic data and predict whether a patient is likely to experience a relapse.

---

##  About Breast Cancer & Relapse

Breast cancer is one of the most prevalent cancers worldwide. While treatments have improved significantly, relapse—defined as the return of cancer after initial recovery—remains a major clinical challenge. Identifying molecular markers that signal relapse risk is essential for improving patient-specific treatment planning. This project explores whether gene expression profiles can be used to predict relapse risk with machine learning.

---

##  Dataset Overview

**Source**: [NCBI GEO – GSE1456](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1456)
 **Samples**: ~159 breast cancer patient tissue samples
**Features**: ~22,000 gene expression values per sample
 **Label Used**: `RELAPSE` (0 = no relapse, 1 = relapse)

The dataset was downloaded and parsed using [`GEOparse`](https://github.com/guma44/GEOparse), a Python library for accessing GEO data.



##  Preprocessing Steps

1. **Gene Filtering**: Genes missing in more than 50% of patients were removed.
2. **Imputation**: Remaining missing values were filled using mean imputation.
3. **Standardization**: Expression values were scaled using `StandardScaler` to normalize the data.
4. **Feature Selection**:
   Low-variance genes removed via `VarianceThreshold`
   Top 50% informative genes selected using `SelectKBest` (ANOVA F-test)



##  Model: Random Forest Classifier

- Trained using `RandomForestClassifier` from Scikit-learn with `class_weight='balanced'` to address class imbalance.
- Data split: 70% training / 30% testing
- Evaluation metrics included accuracy, F1-score, recall, and ROC-AUC.


##  Results & Observations

| Metric           | Result        |

| Accuracy         | 72%           |
| AUC-ROC Score    | ~0.70         |
| F1 Score (class 1 - relapse) | Low (~0.18) due to class imbalance |
| Key Insight      | Model is strong at predicting non-relapse, but weaker at relapse due to fewer positive cases in the dataset |

- **Feature Importance**: Top 20 genes identified by the model can be further explored for biological relevance.
- **PCA Visualization**: PCA reduced data to 2D, showing partial separation between relapse and non-relapse samples.
- **Synthetic Testing**: Simulated patient profiles validated model behavior on unseen examples.


##  Conclusion

This project demonstrates the feasibility of using gene expression profiles to predict relapse in breast cancer patients. While model performance was promising for non-relapse classification, improvements are needed in relapse detection 







##  Ethical Note

- All data used is publicly available and de-identified.
- The model is educational and not intended for clinical use.

---


