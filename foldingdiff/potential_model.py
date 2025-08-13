import os
import tempfile
import subprocess
from typing import List, Optional, Tuple, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import math
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biotite.structure.io.pdb import PDBFile
from Bio.PDB import PDBParser, DSSP
import pickle
from foldingdiff.feats import *

# ---------------------------------------------------------------------
# Semi‑CRF wrapper
# ---------------------------------------------------------------------

class SemiCRFModel(nn.Module):
    """
    Combines:
      – token‑level encoder  (AngleTransformer or BertForDiffusionBase)
      – residue‑wise biochemical features  -> pooled segment vector
      – segment MLP potential
      – optional length bias γ
    The `forward()` returns the *segment score table* out[i][l].
    """
    def __init__(
        self,
        config,
        encoder: Optional[nn.Module] = None,
        length_bias: float = 0.0,
        max_seg_len: int = 100,
        device: Optional[torch.device] = "cpu"
    ):
        super().__init__()
        self.config          = config
        self.encoder         = encoder
        self.featurizer      = BackboneResidueFeaturizer(config, device=device)
        self.aggregator      = SegmentFeatureAggregator(self.featurizer.out_dim, self.featurizer.labels)
        self.seg_mlp         = SegmentPotentialMLP(self.aggregator.out_dim, hidden=64)
        self.gamma           = length_bias
        self.max_seg_len     = max_seg_len
        self.device          = device or torch.device("cpu")
        if self.encoder:
            # linear projection from encoder hidden -> scalar
            self.enc2score = nn.Linear(
                encoder.config.hidden_size if hasattr(encoder, "config") else encoder.d_model,
                1,
            )
        else:
            self.zernike_projector = nn.Linear(9, 1)
            self.feats = defaultdict(dict)
    
    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_fps(args):
        """
        Compute all (i,j) fingerprints for a single protein t.
        Returns (prot_id, { (i,j): torch.Tensor(fp) }).
        """
        t, grid_size, padding, order = args
        prot_id = Path(t.fname).stem
        aa_seq  = t.aa
        coords  = t.compute_coords()
        # ensure numpy for voxelize / 3DZD
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords

        fps_dict = {}
        for i in range(len(aa_seq)):
            for j in range(i+1, len(aa_seq)+1):
                subcoords = coords_np[3*i:3*j]
                grid      = voxelize(subcoords, grid_size=grid_size, padding=padding)
                fp        = compute_3d_zernike(grid, order=order)
                fps_dict[(i, j)] = fp

        return prot_id, fps_dict

    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_plddt(args, batch_size):
        """
        Compute all pLDDT confidences for a batch of proteins.
        Returns (prot_ids, [torch.Tensor(confidences)])
        """
        # 1) Gather IDs (here we assume .aa is the sequence string)
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as in_f:
            for seq in sequences:
                in_f.write(seq+'\n')
            in_path = in_f.name
        print(in_path)
        # 3) Reserve an output path
        out_fd, out_path = tempfile.mkstemp(suffix='.pt')
        os.close(out_fd)
        # locate your script relative to this file
        python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "get_plddt.py")
        try:
            # this will invoke the python inside the esmfold env
            subprocess.run([
                "conda", "run", "-n", "esmfold",
                "python", python_path,
                "--in-file",  in_path,
                "--out-file", out_path,
                "--batch-size", str(batch_size)
            ], check=True)
            plddts = torch.load(out_path)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up
            os.remove(in_path)
            os.remove(out_path)
        print("PLDDT prediction completed.")
        return prot_ids, plddts
    
    @staticmethod
    def _compute_protein_disorder(args):
        """
        Compute all disorder scores for a batch of proteins.
        Returns (prot_ids, [list of disorder scores])
        """       
        # 1) Gather protein IDs and sequences
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences in FASTA format
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.seq', delete=False) as in_f:
            for seq, prot_id in zip(sequences, prot_ids):
                in_f.write(f">{prot_id}\n{seq}\n")
            in_path = in_f.name

        # 3) Path to the iupred2a.py script (modify this according to your setup)
        python_path = os.path.join(Path(__file__).parents[2], "iupred2a/iupred2a.py")
        
        try:
            # 4) Run iupred2a.py using subprocess, capture stdout directly
            result = subprocess.run(
                ["python", python_path, in_path, "long"],  # Assumes "long" mode for disorder prediction
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True  # Capture the output as text
            )            
            # 5) Parse the stdout to extract the per-residue disorder scores
            disorder_scores = []
            lines = result.stdout.split('\n')            
            last = 0
            for seq in sequences:
                scores = []
                for a, l in zip(seq, lines[last: last+len(seq)]):
                    parts = l.split('\t')
                    aa = parts[1]
                    assert a == aa
                    residue_score = float(parts[2])  # The score is in the third column
                    scores.append(residue_score)
                disorder_scores.append(scores)
                last = last + len(seq)
            assert last + 1 == len(lines)

        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up the temporary file
            os.remove(in_path)
        
        print("Disorder prediction completed.")
        return prot_ids, disorder_scores      


    @staticmethod
    def _compute_protein_embedding(args, batch_size):
        """
        Compute all embeddings for a batch of proteins.
        Returns (prot_ids, [list of embeddings scores])
        """       
        # 1) Gather IDs (here we assume .aa is the sequence string)
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as in_f:
            for seq in sequences:
                in_f.write(seq+'\n')
            in_path = in_f.name

        # 3) Reserve an output path
        out_fd, out_path = tempfile.mkstemp(suffix='.pt')
        os.close(out_fd)
        # locate your script relative to this file
        python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "get_embedding.py")
        try:
            # this will invoke the python inside the esmfold env
            subprocess.run([
                "conda", "run", "-n", "esmfold",
                "python", python_path,
                "--in-file",  in_path,
                "--out-file", out_path,
                "--batch-size", str(batch_size)
            ], check=True)
            embeddings = torch.load(out_path)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up
            os.remove(in_path)
            os.remove(out_path)
        print("Embedding done.")
        return prot_ids, embeddings  


    @staticmethod
    def _compute_protein_sec(args):
        """
        Compute all sec features for a batch of proteins.
        Returns (prot_ids, [list of sec types])
        """       
        # 1) Gather protein IDs and sequences
        prot_ids = [Path(t.fname).stem for t in args]
        ss_preds = []
        for t in args:
            fname = t.fname
            parser = PDBParser()
            structure = parser.get_structure(Path(fname).stem, fname)
            import biotite.structure as struc
            source_struct = PDBFile.read(open(t.fname)).get_structure(model=1)
            backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
            res_ids = []
            for a in backbone_atoms:
                if len(res_ids) and res_ids[-1] == a.res_id:
                    continue
                res_ids.append(a.res_id)
            assert len(res_ids) == len(t.aa)
            model = structure[0]  # assuming you want the first model            
            try:
                dssp = DSSP(model, fname)
                print("dssp good")
            except:
                ss_preds.append("".join(["C" for _ in res_ids]))
                continue
            ss_map = {"H":"H", "G":"H", "I":"H",  # helix types → H
                    "E":"E", "B":"E",           # sheet types → E
                    "T":"C", "S":"C", " ": "C", "-": "C"} # turns/others → C
            ss_list = [None for _ in t.aa]
            ss_dict = {}
            for key in dssp.keys():
                _, (_, res_id, _) = key                
                ss_dict[res_id] = ss_map[dssp[key][2]]
            for i, res_id in enumerate(res_ids):
                if res_id in ss_dict:
                    ss_list[i] = ss_dict[res_id]
                else:
                    ss_list[i] = "C"
            ss_pred = "".join(ss_list)  # length L
            assert len(ss_pred) == len(t.aa)
            ss_preds.append(ss_pred)
        print("Secondary structure prediction done.")
        return prot_ids, ss_preds      
    

    def compute_feats(self, dataset, config):
        tasks = {
            "disorder":   lambda: self.compute_disorder(dataset, **config["disorder"]),
            "embeddings": lambda: self.compute_embeddings(dataset, **config["embeddings"]),
            "sec":        lambda: self.compute_sec(dataset, **config["sec"]),
            "plddt":      lambda: self.compute_plddt(dataset, **config["plddt"]),
            "fps":        lambda: self.compute_fps(dataset, **config["fps"]),
        }
        for task in tasks:
            if config[task]["enabled"]:
                tasks[task]()
        
        # spin up one “worker” per compute_*, all running concurrently
        # with ThreadPoolExecutor(max_workers=len(tasks)) as exec:
        #     future_to_name = {
        #         exec.submit(fn): name
        #         for name, fn in tasks.items()
        #     }
        #     for future in as_completed(future_to_name):
        #         name = future_to_name[future]
        #         try:
        #             future.result()   # will re-raise if that task errored
        #             print(f"{name} done")
        #         except Exception as e:
        #             print(f"⚠️ {name} failed: {e}")        

    @staticmethod
    def dump_feat_batch(i, batch_size, save_dir):
        batch_path = os.path.join(save_dir, f"feats_{batch_size}_{i}.pkl")
        print(f"[batch {i}] loading {batch_path!r}")
        with open(batch_path, "rb") as f:
            feats = pickle.load(f)
        for prot_id, feat in feats.items():
            out_path = os.path.join(save_dir, f"{prot_id}.pkl")
            # you can choose "xb" if you want to avoid overwriting existing files
            with open(out_path, "wb") as g:
                pickle.dump(feat, g)
        print(f"[batch {i}] done")
        return i        


    def compute_batch_feats(self, dataset, config, save_dir, batch_size):
        # compute feats in batches    
        n_batches = (len(dataset)+batch_size-1)//batch_size
        for i in range(n_batches):                            
            batch_path = os.path.join(save_dir, f"feats_{batch_size}_{i}.pkl")
            all_dumped = all([os.path.exists(os.path.join(save_dir, f"{Path(t.fname).stem}.pkl")) \
                for t in dataset[batch_size*i:batch_size*(i+1)]])
            if all_dumped:
                print(f"batch {i} already dumped")
                continue            
            elif os.path.exists(batch_path):
                print(f"batch {i} done, continuing")
                continue
            else:
                print(f"begin feat computation batch {i}")
                # check if already all dumped                
                self.compute_feats(dataset[batch_size*i:batch_size*(i+1)], config)
                print(f"begin saving feat batch {i}")
                pickle.dump(self.feats, open(batch_path, "wb+"))
                print(f"done saving feat batch {i}")
                self.feats = defaultdict(dict)
        print("all feat batches ready")
        # cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        # cpus = min(cpus, 10)
        # ctx = mp.get_context('spawn')
        # with ProcessPoolExecutor(max_workers=cpus) as exe:
        #     # This will print each batch’s completion as they finish:
        #     for i in exe.map(SemiCRFModel.dump_feat_batch, range(n_batches), [batch_size]*n_batches, [save_dir]*n_batches):
        #         pass
        for i in range(n_batches):
            all_dumped = all([os.path.exists(os.path.join(save_dir, f"{Path(t.fname).stem}.pkl")) \
                for t in dataset[batch_size*i:batch_size*(i+1)]])
            if all_dumped:
                continue
            else:
                SemiCRFModel.dump_feat_batch(i, batch_size, save_dir)
        print("All feats dumped.")

        
    def compute_fps(self,
                    dataset,
                    grid_size=64,
                    padding=2.0,
                    order=8,
                    max_workers=20,
                    enabled=True):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        args_list = [
            (t, grid_size, padding, order)
            for t in dataset
        ]
        if max_workers:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(SemiCRFModel._compute_protein_fps, args): args[0]
                    for args in args_list
                }

                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing fps"):
                    prot_id, fps_dict = fut.result()
                    self.feats[prot_id]['fps'] = fps_dict
        else:
            res = [SemiCRFModel._compute_protein_fps(args) for args in args_list]
            for prot_id, fps_dict in res:
                self.feats[prot_id]['fps'] = fps_dict                


    def compute_plddt(self,
                    dataset,
                    max_workers=0, 
                    batch_size=1000, 
                    model_batch_size=10,
                    enabled=True):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_plddt, batch_size=model_batch_size)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing plddts"):
                    prot_ids, plddts = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        plddt = plddts[i]
                        plddt = plddt[plddt==plddt]
                        self.feats[prot_id]['plddt'] = plddt.cpu().numpy()
        else:
            prot_ids, plddts = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                plddt = plddts[i]
                plddt[plddt!=plddt] = 0.
                self.feats[prot_id]['plddt'] = plddt.cpu().numpy()               


    def compute_disorder(self,
                    dataset,
                    max_workers=20, 
                    batch_size=5,
                    enabled=True):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_disorder)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')            
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing disorder"):
                    prot_ids, disorders = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['disorder'] = np.array(disorders[i])
        else:
            prot_ids, disorders = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['disorder'] = np.array(disorders[i])                


    def compute_embeddings(self,
                    dataset,
                    max_workers=0, 
                    batch_size=1000, 
                    model_batch_size=10,
                    enabled=True):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_embedding, batch_size=model_batch_size)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            # embeddings = SemiCRFModel._compute_protein_embedding(dataset)        
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing embedding"):
                    prot_ids, embeddings = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['embeddings'] = embeddings[i].cpu().numpy()
        else:
            prot_ids, embeddings = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['embeddings'] = embeddings[i].cpu().numpy()                



    def compute_sec(self,
                    dataset,
                    max_workers=20, 
                    batch_size=20,
                    enabled=True):        
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(SemiCRFModel._compute_protein_sec, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing sec"):
                    prot_ids, secs = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['sec'] = secs[i]
        else:
            prot_ids, secs = SemiCRFModel._compute_protein_sec(dataset)            
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['sec'] = secs[i]                

    # -----------------------------------------------------------------
    # main API ---------------------------------------------------------
    # -----------------------------------------------------------------
    def forward(
        self,
        feats: Dict,
        aa_seq: str,
        angles_tensor: torch.Tensor,            # (1, L, num_features)
        coords_tensor: torch.Tensor,            # (3*L, 3)
        timestep: torch.Tensor,                 # (1,) or (1,1)
        attention_mask: torch.Tensor,           # (1, L)
        batch_size: int = 64
    ) -> List[List[torch.Tensor]]:
        """
        Returns:
            out[i][l]   scalar (log‑)score for span [i, i+l-1]
                        i in 0…L-1, l in 1…L-i
        """

        # 1) token‑level encoder
        #    -> per‑token hidden (batch, L, hidden)
        if self.encoder:
            L = angles_tensor.size(1)
            assert L == len(aa_seq), "angle tensor and sequence length differ"            
            enc_hidden = self.encoder(
                inputs        = angles_tensor.to(self.device),
                timestep      = timestep.to(self.device),
                attention_mask= attention_mask.to(self.device),
            )
            if isinstance(enc_hidden, tuple) or isinstance(enc_hidden, list):
                enc_hidden = enc_hidden[0]          # (1, L, hidden)
            token_repr = enc_hidden.squeeze(0)       # (L, hidden)
        else:
            L = len(aa_seq)
        
        # 2) pre‑compute residue‑level biochemical feature tensor (L, feat_dim)        
        plddt = feats['plddt'] if self.config["plddt"]["enabled"] else None
        disorder = feats['disorder'] if self.config["disorder"]["enabled"] else None
        ss_pred = feats['sec'] if self.config["sec"]["enabled"] else None
        embedding = feats['embeddings'] if self.config["embeddings"]["enabled"] else None    
        res_feats = self.featurizer(
            aa_seq, ss_pred=ss_pred,
            disorder=disorder,
            plddt=plddt,
            embedding=embedding
        ).to(self.device)                        # (L, feat_dim)


        # 3) build out[i][l]
        out: List[List[torch.Tensor]] = [[None]*(L+1) for _ in range(L)]
        attn_out: List[List[torch.Tensor]] = [[None]*(L+1) for _ in range(L)]
        
        # cumulative sums for O(1) span pooling of token repr & res feats
        if self.encoder:
            cumsum_token = torch.cat([torch.zeros(1, token_repr.size(-1), device=self.device),
                                    token_repr.cumsum(dim=0)], dim=0)

        # a) compute bio_scores, in batches
        agg_vecs: List[torch.Tensor] = []
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential
                span_feat = res_feats[i:j]                       # (l, feat_dim)
                agg_vec   = self.aggregator(span_feat)           # (agg_dim,)                
                agg_vecs.append(agg_vec)
                
        # bio_scores = []
        # for i in range((len(agg_vecs)+batch_size-1)//batch_size):
        #     batch = torch.stack(agg_vecs[batch_size*i:batch_size*(i+1)], axis=0)
        #     scores, _ = self.seg_mlp(batch)
        #     bio_scores
        bio_scores, attn_scores = self.seg_mlp(torch.stack(agg_vecs, axis=0))
        index = 0
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential scores
                out[i][l] = bio_scores[index]
                attn_out[i][l] = attn_scores[index]
                index += 1
       
        # b) compute descr_scores
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l

                fp = feats['fps'][(i, j)]
                fp = torch.as_tensor(fp).to(self.device)
                descr_score = self.zernike_projector(fp)

                # total score + length bias
                out[i][l] += descr_score.squeeze()

        # c) add length bias and (if encoder) compute encoder scores
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                if self.encoder:
                    # mean token embedding for span
                    span_token_mean = (cumsum_token[j] - cumsum_token[i]) / l
                    enc_score = self.enc2score(span_token_mean).squeeze(-1)
                else:
                    enc_score = 0.0

                # total score + length bias
                out[i][l] += enc_score - self.gamma

        return out, attn_out

    # convenience helper identical to your old precompute signature
    def precompute(self, feats, aa_seq, angles_tensor=None, coords_tensor=None, timestep=None, attention_mask=None):
        return self.forward(feats, aa_seq, angles_tensor, coords_tensor, timestep, attention_mask)        


class SemiCRF2DModel(SemiCRFModel):
    def __init__(
        self,        
        config,
        length_bias: float = 0.0,
        max_seg_len: int = 100,
        device: Optional[torch.device] = "cpu"
    ):
        super().__init__(
            config=config,
            length_bias=length_bias,
            max_seg_len=max_seg_len,
            device=device
        )
        if config["foldseek"]["enabled"]:
            self.foldseek_projector = nn.Linear(22, 1)
        self.seg_pair_aggregator = SegmentPairFeatureAggregator(10, foldseek_feat_labels)
        self.seg_pair_mlp    = SegmentPairPotentialMLP(self.seg_pair_aggregator.out_dim, hidden=64)                        


    def compute_feats(self, dataset, config):    
        tasks = {
            "disorder":   lambda: self.compute_disorder(dataset, **config["disorder"]),
            "embeddings": lambda: self.compute_embeddings(dataset, **config["embeddings"]),            
            "sec":        lambda: self.compute_sec(dataset, **config["sec"]),
            "plddt":      lambda: self.compute_plddt(dataset, **config["plddt"]),
            "fps":        lambda: self.compute_fps(dataset, **config["fps"]),
            "foldseek":   lambda: self.compute_foldseek(dataset, self.max_seg_len, **config["foldseek"])
        }
        
        for task in tasks:
            if config[task]["enabled"]:
                tasks[task]()
        # spin up one “worker” per compute_*, all running concurrently
        # with ThreadPoolExecutor(max_workers=len(tasks)) as exec:
        #     future_to_name = {
        #         exec.submit(fn): name
        #         for name, fn in tasks.items()
        #     }
        #     for future in as_completed(future_to_name):
        #         name = future_to_name[future]
        #         try:
        #             future.result()   # will re-raise if that task errored
        #             print(f"{name} done")
        #         except Exception as e:
        #             print(f"⚠️ {name} failed: {e}")



    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_foldseeks(args):
        """
        Compute all (i, l1, l2) foldseeks for a single protein t.
        Returns (prot_id, { (i,l1,l2): torch.Tensor(foldseek) }).
        """
        fname, aa_seq, coords, beta_c, max_seg_len = args
        prot_id = Path(fname).stem
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        L = len(aa_seq)
        foldseeks_dict = {}
        for i in range(L):
            max_l2 = min(max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(max_seg_len, i)
                for l1 in range(1, max_l1 + 1):                    
                    c = coords_np[3*(i-l1): 3*(i+l2)]                    
                    cb = beta_c[i-l1:i+l2]
                    n, ca, c = c[0::3], c[1::3], c[2::3]
                    feats, mask, _ = structure2features(ca, n, c, cb)
                    foldseeks_dict[(i, l1, l2)] = feats

        return prot_id, foldseeks_dict        


    def compute_foldseek(self,
                         dataset,
                         max_L,
                         max_workers=20,
                         enabled=True
    ):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        args_list = [
            (t.fname, t.aa, t.compute_coords(), t.beta_coords, max_L)
            for t in dataset
        ]
        if max_workers:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(SemiCRF2DModel._compute_protein_foldseeks, args): args[0]
                    for args in args_list
                }

                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing foldseeks"):
                    prot_id, foldseeks_dict = fut.result()
                    self.feats[prot_id]['foldseek'] = foldseeks_dict
        else:
            res = [SemiCRF2DModel._compute_protein_foldseeks(args) for args in args_list]
            for prot_id, foldseeks_dict in res:
                self.feats[prot_id]['foldseek'] = foldseeks_dict     
    
    # -----------------------------------------------------------------
    # main API ---------------------------------------------------------
    # -----------------------------------------------------------------
    def forward(
        self,
        feats: Dict,
        aa_seq: str
    ) -> Tuple[
        torch.Tensor,                              # unary_out, shape (L, max_seg_len+1)
        List[List[torch.Tensor]],                  # unary_attn_out, shape (L, max_seg_len+1)
        torch.Tensor,                              # edge_out, shape (L, max_seg_len+1, max_seg_len+1)
        List[List[List[torch.Tensor]]]             # edge_attn_out, shape (L, max_seg_len+1, max_seg_len+1)
    ]:
        """
        Compute unary and pairwise segment scores for a protein sequence.
        Args:
            feats (Dict): Dictionary of per-protein features, including 'plddt', 'disorder', 'sec', 'embeddings', 'fps', 'foldseek'.
            aa_seq (str): Amino acid sequence string for the protein.

        Returns:
            unary_out (torch.Tensor): Segment unary scores, shape (L, max_seg_len+1), where unary_out[i][l] is the score for span [i, i+l-1].
            unary_attn_out (List[List[torch.Tensor]]): Unary attention weights per span [[attn_vec for l in 0..max_seg_len] for i in 0..L-1].
            edge_out (torch.Tensor): Pairwise segment scores, shape (L, max_seg_len+1, max_seg_len+1), where edge_out[i][l1][l2] scores split at i.
            edge_attn_out (List[List[List[torch.Tensor]]]): Pairwise attention weights per span/split [[[attn_vec for l2] for l1] for i].
        """
        # 1) token‑level encoder
        #    -> per‑token hidden (batch, L, hidden)
        L = len(aa_seq)
        
        # 2) pre‑compute residue‑level biochemical feature tensor (L, feat_dim)        
        plddt = feats['plddt'] if self.config["plddt"]["enabled"] else None
        disorder = feats['disorder'] if self.config["disorder"]["enabled"] else None
        ss_pred = feats['sec'] if self.config["sec"]["enabled"] else None
        embedding = feats['embeddings'] if self.config["embeddings"]["enabled"] else None
        res_feats = self.featurizer(
            aa_seq, ss_pred=ss_pred,
            disorder=disorder,
            plddt=plddt,
            embedding=embedding
        ).to(self.device)                        # (L, feat_dim)

        # 3) build unary_out[i][l] and edge_out[i][l1][l2]
        unary_out: torch.Tensor = torch.tensor([[0.0]*(self.max_seg_len+1) for _ in range(L)], device=self.device)
        unary_attn_out: List[List[torch.Tensor]] = [[0.0]*(self.max_seg_len+1) for _ in range(L)]
        edge_out: torch.Tensor = torch.tensor([[[0.0]*(self.max_seg_len+1)]*(self.max_seg_len+1) for _ in range(L)], device=self.device)
        edge_attn_out: List[List[List[torch.Tensor]]] = [[[0.0]*(self.max_seg_len+1)]*(self.max_seg_len+1) for _ in range(L)]
        
        # a) compute bio_scores, in batches
        agg_vecs: List[torch.Tensor] = []
        agg_vec_pairs: List[List[torch.Tensor]] = []
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential
                span_feat = res_feats[i:j]                       # (l, feat_dim)
                agg_vec   = self.aggregator(span_feat)           # (agg_dim,)                
                agg_vecs.append(agg_vec)
        
        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    # biochemical segment potential
                    span_feat1 = self.aggregator(res_feats[i-l1: i])
                    span_feat2 = self.aggregator(res_feats[i: i+l2])
                    pair_span_feat = torch.cat((span_feat1, span_feat2))
                    agg_vec_pairs.append(pair_span_feat)

        bio_scores, attn_scores = self.seg_mlp(torch.stack(agg_vecs, axis=0))
        bio_pair_scores, attn_pair_scores = self.seg_pair_mlp(torch.stack(agg_vec_pairs, axis=0))

        index = 0
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential scores
                unary_out[i][l] = bio_scores[index]
                unary_attn_out[i][l] = attn_scores[index]
                index += 1

        index = 0
        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    # biochemical segment potential scores
                    edge_out[i][l1][l2] = bio_pair_scores[index]
                    edge_attn_out[i][l1][l2] = attn_pair_scores[index]
                    index += 1
       
        # b) compute descr_scores
        if self.config["fps"]["enabled"]:
            for i in range(L):
                max_l = min(self.max_seg_len, L - i)
                for l in range(1, max_l + 1):
                    j = i + l
                    fp = feats['fps'][(i, j)]
                    fp = torch.as_tensor(fp, dtype=torch.float32).to(self.device)
                    descr_score = self.zernike_projector(fp)
                    # total score
                    unary_out[i][l] += descr_score.squeeze()

        if self.config["foldseek"]["enabled"]:
            for i in range(L):
                max_l2 = min(self.max_seg_len, L - i)
                for l2 in range(1, max_l2 + 1):
                    max_l1 = min(self.max_seg_len, i)
                    for l1 in range(1, max_l1 + 1):
                        foldseek = feats['foldseek'][(i, l1, l2)]
                        foldseek = torch.as_tensor(foldseek, dtype=torch.float32).to(self.device)
                        foldseek = self.seg_pair_aggregator(foldseek)
                        descr_score = self.foldseek_projector(foldseek)
                        edge_out[i][l1][l2] += descr_score.squeeze()

        # c) add length bias
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                unary_out[i][l] -= self.gamma

        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    edge_out[i][l1][l2] -= self.gamma

        return unary_out, unary_attn_out, edge_out, edge_attn_out


    # convenience helper identical to your old precompute signature
    def precompute(self, feats, aa_seq):
        return self.forward(feats, aa_seq)   
    