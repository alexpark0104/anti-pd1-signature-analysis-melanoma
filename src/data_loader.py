import pandas as pd
import numpy as np
import gzip

def load_discovery_data(expr_path, clin_path):
    expr_raw = pd.read_csv(expr_path, index_col=0)
    expr = expr_raw.T
    clinical = pd.read_csv(clin_path, index_col=0)

    def format_sample(s):
        parts = s.split('_')
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1].capitalize()}"
        return s

    clinical['expr_sample'] = clinical['Sample'].apply(format_sample)
    common = set(clinical['expr_sample']).intersection(expr.index)
    expr = expr.loc[list(common)]
    clinical = clinical.set_index('expr_sample').loc[list(common)]

    def map_response(val):
        if isinstance(val, str):
            v = val.lower()
            if 'cr' in v or 'pr' in v:
                return 1
            elif 'sd' in v or 'pd' in v:
                return 0
        return np.nan

    labels_raw = clinical['myBOR'].map(map_response)
    valid = labels_raw.notna()
    return expr.loc[valid], labels_raw[valid], clinical.loc[valid]

def parse_gse78220_series_matrix(raw_text):
    lines = raw_text.split('\n')
    gsm_list, pt_list, labels = [], [], {}
    for line in lines:
        if line.startswith('!Sample_geo_accession'):
            gsm_list = [p.strip().strip('"').strip("'") for p in line.split('\t')[1:] if p.strip()]
        if line.startswith('!Sample_title'):
            pt_list = [p.strip().strip('"').strip("'") for p in line.split('\t')[1:] if p.strip()]
    if not gsm_list or not pt_list:
        raise ValueError("No sample IDs")
    gsm_to_pt = dict(zip(gsm_list, pt_list))
    for line in lines:
        if line.startswith('!Sample_source_name_ch1'):
            vals = [p.strip().lower() for p in line.split('\t')[1:] if p.strip()]
            for i, v in enumerate(vals):
                if i < len(gsm_list):
                    gsm = gsm_list[i]
                    if 'partial response' in v:
                        labels[gsm] = 1
                    elif 'progressive disease' in v or 'stable disease' in v:
                        labels[gsm] = 0
    if not labels:
        for line in lines:
            if 'anti-pd-1 response' in line and line.startswith('!Sample_characteristics_ch1'):
                vals = [p.strip().lower() for p in line.split('\t')[1:] if p.strip()]
                for i, v in enumerate(vals):
                    if i < len(gsm_list):
                        gsm = gsm_list[i]
                        if 'partial response' in v:
                            labels[gsm] = 1
                        elif 'progressive disease' in v or 'stable disease' in v:
                            labels[gsm] = 0
    labels_pt = {}
    for gsm, label in labels.items():
        pt = gsm_to_pt.get(gsm)
        if pt:
            labels_pt[pt] = label
    return pd.Series(labels_pt).dropna()

def load_validation_data(series_path, expr_path):
    with open(series_path, 'rb') as f:
        raw = gzip.decompress(f.read()).decode('utf-8', errors='ignore')
    labels = parse_gse78220_series_matrix(raw)
    expr_raw = pd.read_excel(expr_path, index_col=0)
    expr = expr_raw.T
    expr.index = [idx.split('.')[0] for idx in expr.index]
    common = expr.index.intersection(labels.index)
    return expr.loc[common], labels.loc[common]
