import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
from config import ENTREZ_TO_SYMBOL, COEFFICIENTS

def compute_signature_score(expr, gene_symbols, coefficients):
    present_symbols, weights = [], []
    for sym, coef in zip(gene_symbols, coefficients):
        if sym in expr.columns:
            present_symbols.append(sym); weights.append(coef)
        else:
            matches = [c for c in expr.columns if sym.upper() in c.upper()]
            if matches:
                present_symbols.append(matches[0]); weights.append(coef)
    if len(present_symbols) < 5:
        raise ValueError(f"Too few signature genes available: {len(present_symbols)}")
    X_sig = expr[present_symbols].values
    return X_sig @ np.array(weights), present_symbols

def validate_on_gse78220(expr, labels, gene_symbols, coefficients, save_plot=True):
    score, present = compute_signature_score(expr, gene_symbols, coefficients)
    y_true = labels.values
    if expr.values.max() > 100:
        X_sig_log = np.log2(expr[present].values + 1)
        score_log = X_sig_log @ np.array(coefficients[:len(present)])
        auc = roc_auc_score(y_true, score_log)
    else:
        auc = roc_auc_score(y_true, score)
    if save_plot:
        fpr, tpr, _ = roc_curve(y_true, score)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2.5, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation on GSE78220 (Hugo et al.)')
        plt.legend()
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/validation_roc.png', dpi=300)
        plt.close()
    return auc, present
