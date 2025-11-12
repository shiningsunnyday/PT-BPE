import pickle
from pathlib import Path
import csv
import argparse
import os
import json
import pandas as pd
import numpy as np
import random
from collections import Counter
from tqdm import tqdm


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


def _convert_true_intervals(true_domains):
    """
    Convert true domain matches from 1-based inclusive to 0-based half-open intervals.
    Input: list of (from_residue, to_residue) inclusive, 1-based.
    Returns: list of (start_idx, end_idx) where start_idx inclusive, end_idx exclusive.
    """
    return [(f - 1, t) for f, t in true_domains]

def _best_block_set(f, t, pred_segs):
    """
    Find the best consecutive set of predicted segments overlapping [f, t)
    by maximizing IoU. Returns:
      (start_idx, end_idx, best_recall, best_precision, best_iou)
    where start_idx/end_idx are indices into pred_segs (inclusive).
    """
    idxs = [i for i, (p, q) in enumerate(pred_segs) if q > f and p < t]
    if not idxs:
        return None, None, 0.0, 0.0, 0.0

    i0, i1 = min(idxs), max(idxs)
    best_set = (None, None)
    best_recall = best_precision = best_iou = 0.0
    domain_len = t - f

    # Try trimming 0 or 1 segment at each end
    for trim_left in (0, 1):
        for trim_right in (0, 1):
            a = i0 + trim_left
            b = i1 - trim_right
            if a is None or b is None or a > b:
                continue

            chosen = pred_segs[a:b+1]
            overlap = sum(max(0, min(t, q) - max(f, p)) for p, q in chosen)
            pred_len = sum((q - p) for p, q in chosen)
            recall = overlap / domain_len if domain_len > 0 else 0.0
            precision = overlap / pred_len if pred_len > 0 else 0.0
            union = domain_len + pred_len - overlap
            iou = overlap / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_recall = recall
                best_precision = precision
                best_set = (a, b)

    return best_set[0], best_set[1], best_recall, best_precision, best_iou

def domain_f1(true_domains, pred_segs, iou_threshold=0.5):
    """
    For each true domain, match to the best set of consecutive predicted segments,
    then compute per-domain recall, precision, IoU, and their means,
    plus overall segment-level Precision/Recall/F1 at IoU threshold.
    """
    true_iv = _convert_true_intervals(true_domains)
    recs, precs, ious = [], [], []
    matched_true = 0
    matched_pred_idxs = set()

    for (f, t) in true_iv:
        a, b, r, p, iou = _best_block_set(f, t, pred_segs)
        recs.append(r)
        precs.append(p)
        ious.append(iou)
        if iou >= iou_threshold and a is not None:
            matched_true += 1
            matched_pred_idxs.update(range(a, b+1))

    # Per-domain means
    mean_rec = np.mean(recs) if recs else 0.0
    mean_prec = np.mean(precs) if precs else 0.0
    mean_f1 = np.mean([2 * r * p / (r + p) if (r + p) > 0 else 0.0 for r, p in zip(recs, precs)]) if recs else 0.0
    mean_iou = np.mean(ious) if ious else 0.0

    # Segment-level metrics
    seg_prec = len(matched_pred_idxs) / len(pred_segs) if pred_segs else 0.0
    seg_rec = matched_true / len(true_domains) if true_domains else 0.0
    seg_f1 = (2 * seg_prec * seg_rec / (seg_prec + seg_rec)) if (seg_prec + seg_rec) > 0 else 0.0

    return {
        "mean_recall": mean_rec,
        "mean_precision": mean_prec,
        "mean_f1": mean_f1,
        "mean_iou": mean_iou,
        "segment_precision": seg_prec,
        "segment_recall": seg_rec,
        "segment_f1": seg_f1,
    }

def boundary_metrics(true_domains, pred_segs, delta=0):
    """
    As before: boundary Precision/Recall/F1 with tolerance delta.
    """
    true_intervals = _convert_true_intervals(true_domains)
    true_bounds = set()
    for (f, t) in true_intervals:
        true_bounds.update([f, t])
    pred_bounds = set()
    for (p, q) in pred_segs:
        pred_bounds.update([p, q])

    matched_t = sum(any(abs(tb - pb) <= delta for pb in pred_bounds) for tb in true_bounds)
    matched_p = sum(any(abs(pb - tb) <= delta for tb in true_bounds) for pb in pred_bounds)
    prec = matched_p / len(pred_bounds) if pred_bounds else 0.0
    rec  = matched_t / len(true_bounds) if true_bounds else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def random_partition(seq_len, num_segments):
    """
    Create a random partition of [0, seq_len) into num_segments half-open intervals.
    """
    # choose num_segments-1 cut points between 1 and seq_len-1
    if num_segments <= 1:
        return [(0, seq_len)]
    cuts = sorted(random.sample(range(1, seq_len), num_segments - 1))
    segments = []
    prev = 0
    for c in cuts:
        segments.append((prev, c))
        prev = c
    segments.append((prev, seq_len))
    return segments


def main(args):
    bpe = pickle.load(open(args.pkl_file, 'rb'))
    all_metrics = []
    sorted_tokenizers = sorted(bpe.tokenizers, key=lambda t: Path(t.fname).stem)
    for i in range(len(sorted_tokenizers)):
        t = sorted_tokenizers[i]
        p = Path(t.fname)
        # prot_id = p.stem
        # path = os.path.join(save_dir, f"{prot_id}.json")
        # if os.path.exists(path):
        #     print(f"eval {t.fname}")
        #     vals = json.load(open(path)).values()
        # else:
        #     continue
        vals = t.bond_to_token.values()
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
            for (start, _, l) in vals:
                if l % 3 != 0:
                    assert start + l == 3*t.n-1
                    assert l % 3 == 2
                    l += 1
                pred_segs.append((start//3, start//3+l//3))     
        else:
            continue
        
        true_domains = [(f, to) for f, to in df[['ali_from', 'ali_to']].values if (f > 1 or to < t.n-1) and (to-f+1 <= args.max_len)] # ignore whole-protein is one domain situations
        if not true_domains:
            continue

        # 1) Compute observed metrics
        metrics = {
            "name": p.name,
            "n": len(true_domains)
        }
        # boundary_... keys
        metrics.update({f"boundary_{k}": v 
                        for k, v in boundary_metrics(true_domains, pred_segs, delta=0).items()})
        # domain-level metrics
        metrics.update(domain_f1(true_domains, pred_segs, iou_threshold=0.5))

        # 2) Prepare for randomization
        seq_len = pred_segs[-1][1]
        num_segments = len(pred_segs)
        # numeric metric keys (exclude name,n)
        metric_keys = [k for k in metrics if k not in ("name", "n")]

        # 3) Generate random partitions & collect metric distributions
        random_stats = {k: [] for k in metric_keys}
        N = 1000
        for _ in tqdm(range(N), "scoring random partitions"):
            rand_segs = random_partition(seq_len, num_segments)
            # compute the same metrics on the random segmentation
            b_stats = boundary_metrics(true_domains, rand_segs, delta=0)
            d_stats = domain_f1(true_domains, rand_segs, iou_threshold=0.5)
            # record boundary_... keys
            for k, v in b_stats.items():                
                random_stats[f"boundary_{k}"].append(v)
            # record domain-level keys
            for k, v in d_stats.items():                
                random_stats[k].append(v)

        # 4) Compute one-sided p-values (greater-or-equal)
        for k in metric_keys:
            obs = metrics[k]
            vals = random_stats[k]
            # +1 correction for zero counts
            p_val = (sum(obs >= v for v in vals) + 1) / (N + 1)            
            metrics[f"{k}_pval"] = p_val
        all_metrics.append(dict(sorted(metrics.items())))
        if len(all_metrics) == 5:
            break

    # After loop, all_metrics has p-values alongside your existing metrics.

    rv = lambda v: round(100*v, 2)    
    df = pd.DataFrame(all_metrics)
    # for c in df.columns:
    #     if c in ['name', 'n']:
    #         continue
    #     print(c, rv(df[c].mean()))
    
    for c in ['mean_recall', 'mean_precision', 'mean_f1', 'mean_iou', 'segment_recall', 'segment_precision', 'segment_f1', 'boundary_recall', 'boundary_precision', 'boundary_f1']:
        avg = rv((df[c]*df['n']).sum()/df['n'].sum())
        p_val = rv((df[c+"_pval"]*df['n']).sum()/df['n'].sum())
        print(c, f"{avg} ({p_val})")
        


# def main(args):
#     bpe = pickle.load(open(args.pkl_file, 'rb'))
#     all_metrics = []
#     for i in range(len(bpe.tokenizers)):
#         t = bpe.tokenizers[i]
#         p = Path(t.fname)
#         r = p.relative_to(os.getcwd())
#         n = Path(p.name)
#         out = Path(os.path.join('./scripts/', r, n.with_suffix('.domtblout')))
#         if os.path.exists(out):
#             csv_out = out.with_suffix(".csv")
#             try:
#                 parse_and_write(out, csv_out)
#             except:
#                 print(out)
#                 continue
#             # parse_crh(out, csv_out)
#             df = pd.read_csv(csv_out)
#             pred_segs = []
#             for (start, _, l) in t.bond_to_token.values():
#                 if l % 3 != 0:
#                     assert start + l == 3*t.n-1
#                     assert l % 3 == 2
#                     l += 1
#                 pred_segs.append((start//3, start//3+l//3))                        
#             true_domains = []
#             for (f, to) in df[['ali_from', 'ali_to']].values:
#                 true_domains.append((f, to))
#             if len(true_domains) == 0:
#                 continue
#             metrics = {
#                 "name": p.name, 
#                 "n": len(true_domains)
#             }
#             metrics.update({
#                 f"boundary_{k}": v for (k, v) in boundary_metrics(true_domains, pred_segs, delta=0).items()
#             })
#             metrics.update(domain_f1(true_domains, pred_segs, iou_threshold=0.5))
#             # TODO: generate 10000 random segmentations, compute metrics, and record p-values
#             all_metrics.append(metrics)

#     df = pd.DataFrame(all_metrics)
#     for c in df.columns[2:]:
#         print(c, (df[c]*df['n']).sum()/df['n'].sum())    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_file")
    parser.add_argument("--max_len", default=1000, type=int, help="max len true domains to consider")
    args = parser.parse_args()
    main(args)    