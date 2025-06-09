import pickle
from pathlib import Path
import csv
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, adjusted_rand_score, mutual_info_score
import math
from collections import Counter





def parse_and_write(infile, outfile):
    records = []
    with open(infile) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            # first 22 columns
            (
                target_name, target_acc, tlen,
                query_name, query_acc, qlen,
                e_value, score, bias,
                dom_num, dom_of,
                dom_c_evalue, dom_i_evalue,
                dom_score, dom_bias,
                dom_from, dom_to,
                ali_from, ali_to,
                env_from, env_to,
                acc
            ) = parts[:22]
            description = ' '.join(parts[22:])

            records.append({
                'target_name':      target_name,
                'target_accession': target_acc,
                'tlen':             int(tlen),
                'query_name':       query_name,
                'query_accession':  query_acc,
                'qlen':             int(qlen),
                'E_value':          float(e_value),
                'score':            float(score),
                'bias':             float(bias),
                'domain_num':       int(dom_num),
                'domain_of':        int(dom_of),
                'dom_c_Evalue':     float(dom_c_evalue),
                'dom_i_Evalue':     float(dom_i_evalue),
                'dom_score':        float(dom_score),
                'dom_bias':         float(dom_bias),
                'dom_from':         int(dom_from),
                'dom_to':           int(dom_to),
                'ali_from':         int(ali_from),
                'ali_to':           int(ali_to),
                'env_from':         int(env_from),
                'env_to':           int(env_to),
                'acc':              float(acc),
                'description':      description
            })

    fieldnames = [
        'target_name','target_accession','tlen',
        'query_name','query_accession','qlen',
        'E_value','score','bias',
        'domain_num','domain_of',
        'dom_c_Evalue','dom_i_Evalue','dom_score','dom_bias',
        'dom_from','dom_to',
        'ali_from','ali_to',
        'env_from','env_to',
        'acc','description'
    ]
    with open(outfile, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_crh(inpath, outpath):
    with open(inpath, 'r') as f:
        field_line = None
        for line in f:
            if line.startswith('#FIELDS'):
                # e.g. "#FIELDS query-id match-id score boundaries resolved cond-evalue indp-evalue"
                field_line = line.lstrip('#FIELDS').strip().split()
                break
        if field_line is None:
            raise RuntimeError("No #FIELDS line found in input")

        # We'll expand these two into _from/_to columns
        expand_ranges = ['boundaries', 'resolved']

        # Build output fieldnames: for each in field_line:
        #  - if in expand_ranges, replace with two fields X_from, X_to
        #  - else use the original name (normalized)
        out_fields = []
        for fn in field_line:
            if fn in expand_ranges:
                out_fields += [f"{fn}_from", f"{fn}_to"]
            else:
                # normalize hyphens to underscores
                out_fields.append(fn.replace('-', '_'))

        # Append numeric conversions for clarity (score, cond_evalue, indp_evalue)
        # They already appear in out_fields as strings; converting happens row-wise.

        # Rewind to beginning for actual parsing
        f.seek(0)

        records = []
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != len(field_line):
                raise RuntimeError(f"Line has {len(parts)} cols but expected {len(field_line)}: {line}")

            row = dict(zip(field_line, parts))

            out = {}
            for fn in field_line:
                val = row[fn]
                if fn in expand_ranges:
                    start, end = val.split('-', 1)
                    out[f"{fn}_from"] = int(start)
                    out[f"{fn}_to"]   = int(end)
                elif fn == 'score':
                    out['score'] = float(val)
                elif fn == 'cond-evalue':
                    out['cond_evalue'] = float(val)
                elif fn == 'indp-evalue':
                    out['indp_evalue'] = float(val)
                else:
                    out[fn.replace('-', '_')] = val

            records.append(out)

    # Write CSV
    with open(outpath, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(records)


def _entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count/total) * math.log(count/total) for count in counts.values())

def _convert_true_intervals(true_domains):
    """
    Convert true domain matches from 1-based inclusive to 0-based half-open intervals.
    true_domains: list of (from_residue, to_residue) inclusive, 1-based.
    Returns: list of (start_idx, end_idx) where start_idx inclusive, end_idx exclusive.
    """
    return [(f - 1, t) for f, t in true_domains]

def convert_true_labels(true_domains, seq_len):
    """
    Label each residue by true domain index (1..N) or 0 for background.
    true_domains: list of (from_residue, to_residue) inclusive, 1-based.
    """
    labels = np.zeros(seq_len, dtype=int)
    for idx, (f, t) in enumerate(true_domains):
        start, end = f - 1, t  # convert to half-open
        labels[start:end] = idx + 1
    return labels

def convert_pred_labels(pred_segs, seq_len):
    """
    Label each residue by predicted segment index (1..M).
    pred_segs: list of (start_idx, end_idx) half-open, 0-based.
    """
    labels = np.zeros(seq_len, dtype=int)
    for idx, (start, end) in enumerate(pred_segs):
        labels[start:end] = idx + 1
    return labels

def domain_coverage(true_domains, pred_segs, thresholds=(0.5, 0.8)):
    """
    Compute coverage per true domain:
    For each true domain (1-based inclusive), convert to 0-based half-open,
    find the predicted segment with maximal overlap, then coverage = overlap / true_length.
    """
    true_intervals = _convert_true_intervals(true_domains)
    coverages = []
    for f, t in true_intervals:
        true_len = t - f  # correct length for half-open
        best_overlap = 0
        for p, q in pred_segs:
            overlap = max(0, min(t, q) - max(f, p))
            best_overlap = max(best_overlap, overlap)
        coverages.append(best_overlap / true_len)
    mean_cov = np.mean(coverages)
    recall_at = {f"recall@{int(th*100)}": np.mean([c >= th for c in coverages]) for th in thresholds}
    return {"mean_coverage": mean_cov, **recall_at}

def iou_metrics(true_domains, pred_segs, iou_threshold=0.5):
    """
    Compute IoU-based Precision/Recall/F1:
    Match each true domain to predicted segment with highest IoU.
    """
    true_intervals = _convert_true_intervals(true_domains)
    matched_true = 0
    matched_pred = set()
    for f, t in true_intervals:
        best_iou = 0
        best_j = None
        for j, (p, q) in enumerate(pred_segs):
            inter = max(0, min(t, q) - max(f, p))
            union = (t - f) + (q - p) - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            matched_true += 1
            matched_pred.add(best_j)
    precision = len(matched_pred) / len(pred_segs) if pred_segs else 0
    recall = matched_true / len(true_domains) if true_domains else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def per_residue_metrics(true_domains, pred_segs, seq_len):
    """
    Compare partitions as clusterings at residue level:
    - ARI and VI require labeling each residue by its domain/segment index.
    """
    # Generate label arrays
    true_labels = convert_true_labels(true_domains, seq_len)
    pred_labels = convert_pred_labels(pred_segs, seq_len)

    # Overall clustering agreement
    rand_idx = adjusted_rand_score(true_labels, pred_labels)
    mi = mutual_info_score(true_labels, pred_labels)
    H_true = _entropy(true_labels)
    H_pred = _entropy(pred_labels)
    vi = H_true + H_pred - 2 * mi

    return {
        "rand_index": rand_idx,
        "variation_of_information": vi
    }

def boundary_metrics(true_domains, pred_segs, delta=0):
    """
    Boundary Precision/Recall/F1 with tolerance delta:
    Convert true to half-open for boundary positions, compare starts and ends.
    """
    true_intervals = _convert_true_intervals(true_domains)
    true_bounds = set()
    for f, t in true_intervals:
        true_bounds.update([f, t])
    pred_bounds = set()
    for p, q in pred_segs:
        pred_bounds.update([p, q])

    matched_true = sum(any(abs(tb - pb) <= delta for pb in pred_bounds) for tb in true_bounds)
    matched_pred = sum(any(abs(pb - tb) <= delta for tb in true_bounds) for pb in pred_bounds)
    precision = matched_pred / len(pred_bounds) if pred_bounds else 0
    recall = matched_true / len(true_bounds) if true_bounds else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}



def main(args):
    bpe = pickle.load(open(args.pkl_file, 'rb'))
    all_metrics = []
    for i in range(len(bpe.tokenizers)):
        t = bpe.tokenizers[i]
        p = Path(t.fname)
        r = p.relative_to(os.getcwd())
        n = Path(p.name)
        out = Path(os.path.join('./scripts/', r, n.with_suffix('.domtblout')))
        if os.path.exists(out):
            csv_out = out.with_suffix(".csv")
            try:
                parse_and_write(out, csv_out)
            except:
                print(out)
                continue
            # parse_crh(out, csv_out)
            df = pd.read_csv(csv_out)
            pred_segs = []
            for (start, _, l) in t.bond_to_token.values():
                if l % 3 != 0:
                    assert start + l == 3*t.n-1
                    assert l % 3 == 2
                    l += 1
                pred_segs.append((start//3, start//3+l//3))
            true_domains = []
            for (f, to) in df[['ali_from', 'ali_to']].values:
                true_domains.append((f, to))
            if len(true_domains) == 0:
                continue
            metrics = {"name": p.name, 
                    "domain_coverage": domain_coverage(true_domains, pred_segs),
                    "iou": iou_metrics(true_domains, pred_segs), 
                    "boundary": boundary_metrics(true_domains, pred_segs, delta=0),
                    "n": len(true_domains)
            }
            for m in ["domain_coverage", "iou", "boundary"]:
                if not isinstance(metrics[m], dict):
                    continue
                for k in metrics[m]:
                    metrics[m+"_"+k] = metrics[m][k]
                metrics.pop(m)
            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    for c in df:
        metric_col = False
        for n in ["domain_coverage", "iou", "boundary"]:
            if c[:len(n)] == n:
                metric_col = True
        if metric_col:
            print(c, (df[c]*df['n']).sum()/df['n'].sum())    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_file")
    args = parser.parse_args()
    main(args)    