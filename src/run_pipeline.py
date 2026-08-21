import os, pickle, pandas as pd, numpy as np
from data_loader import load_discovery_data, load_validation_data
from analysis import univariate_screening, nested_cv
from validation import validate_on_gse78220
from config import SIGNATURE_GENES, COEFFICIENTS, ENTREZ_TO_SYMBOL

os.makedirs('results', exist_ok=True)
print("Loading discovery data...")
expr, labels, _ = load_discovery_data('data/rld.BMS038.20171011.csv', 'data/bms038_clinical_data.csv')
X, y = expr.values, labels.values
print(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}")

print("Running nested 5-fold CV...")
mean_auc, overall_auc, _ = nested_cv(X, y, n_genes=10)
print(f"Mean AUC: {mean_auc:.3f}, Overall AUC: {overall_auc:.3f}")

print("Running external validation...")
try:
    expr_val, labels_val = load_validation_data('data/GSE78220_series_matrix.txt.gz', 'data/GSE78220_PatientFPKM.xlsx')
    auc_val, _ = validate_on_gse78220(expr_val, labels_val, list(ENTREZ_TO_SYMBOL.values()), COEFFICIENTS)
    print(f"Validation AUC: {auc_val:.3f}")
except Exception as e:
    print(f"Validation failed: {e}")
