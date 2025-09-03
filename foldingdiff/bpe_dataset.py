import os
import time
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from Bio import PDB
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from foldingdiff.datasets import extract_pdb_code_and_chain

from typing import List
import numpy as np

from esm.utils.structure.protein_chain import ProteinChain
from biotite.structure.io.pdbx import CIFFile, convert
import biotite.structure as bs
from Bio.Data import PDBData
from esm.utils import residue_constants as RC
import torch

from esm.utils.structure.normalize_coordinates import normalize_coordinates

import io
from pathlib import Path
from cloudpathlib import CloudPath
from typing import Sequence, TypeVar, Union
PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import Dataset
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

gpu_lock = threading.Lock()
client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"

def pad_structures(items, constant_value=0, dtype=None, truncation_length=600, pad_length=None):
    """Reference to TAPE https://github.com/songlab-cal/tape/blob/6d345c2b2bbf52cd32cf179325c222afd92aec7e/tape/datasets.py#L37
    """
    batch_size = len(items)
    if isinstance(items[0], List):
        items = [torch.tensor(x) for x in items]
    if pad_length is None:
        shape = [batch_size] + np.max([x.shape for x in items], 0).tolist()
    else:
        shape = [batch_size] + [pad_length]
    if shape[1] > truncation_length:
        shape[1] = truncation_length

    if dtype is None:
        dtype = items[0].dtype

    if isinstance(items[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(items[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, x in zip(array, items):
        arrslice = tuple(slice(dim) for dim in x.shape)
        arr[arrslice] = x[:truncation_length]

    return array


class WrappedProteinChain(ProteinChain):

    """Enable cif file loading, similar to loading pdb.
    Reference to from_pdb in https://github.com/evolutionaryscale/esm/blob/f342784d6a4a5488bfb6c9548530d9724531c85c/esm/utils/structure/protein_chain.py#L539
    """

    @classmethod
    def from_cif_list(
        cls,
        path: PathOrBuffer,
        chain_id_list: list,
        id: str | None = None,
        is_predicted: bool = False,
    ) -> list:

        atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"])
        ret = []
        for chain_id in chain_id_list:
            try:
                pdb_chain = cls.from_cif(path, chain_id, id, is_predicted, atom_array)
            except:
                print(f"Cannot retrieve from local cluster", id, chain_id)
                pdb_chain = None
            ret.append(pdb_chain)
        
        return ret

    @classmethod
    def from_cif(
        cls,
        path: PathOrBuffer,
        chain_id: str = "detect",
        id: str | None = None,
        is_predicted: bool = False,
        atom_array=None,
    ) -> "ProteinChain":
        """Return a ProteinStructure object from a cif file.
        """

        if id is not None:
            file_id = id
        else:
            if isinstance(path, (Path, str)):
                file_id = Path(path).with_suffix("").name
            else:
                file_id = "null"
        
        if atom_array is None:
            atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"])
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        if not (atom_array.chain_id == chain_id).any():
            atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"], use_author_fields=False)

        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        entity_id = 1  # Not supplied in PDBfiles

        sequence = "".join(
            (
                r
                if len(r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X"))
                == 1
                else "X"
            )
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, RC.atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        atom_mask = np.full(
            [num_res, RC.atom_type_num],
            False,
            dtype=bool,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)
        insertion_code = np.full([num_res], "", dtype="<U4")

        confidence = np.ones(
            [num_res],
            dtype=np.float32,
        )

        for i, res in enumerate(bs.residue_iter(atom_array)):
            chain = atom_array[atom_array.chain_id == chain_id]
            assert isinstance(chain, bs.AtomArray)

            res_index = res[0].res_id
            residue_index[i] = res_index
            insertion_code[i] = res[0].ins_code

            # Atom level features
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    atom_name = "SD"

                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                    atom_mask[i, RC.atom_order[atom_name]] = True
                    if is_predicted and atom_name == "CA":
                        confidence[i] = atom.b_factor

        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
        )
        

    def to_structure_encoder_inputs(
        self,
        device="cpu",
        should_normalize_coordinates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.tensor(self.atom37_positions, dtype=torch.float32, device=device)
        plddt = torch.tensor(self.confidence, dtype=torch.float32, device=device)
        residue_index = torch.tensor(self.residue_index, dtype=torch.long, device=device)

        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)
    

def convert_chain_id(pdb_path, chain_id):

    if pdb_path.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
    
    structure = parser.get_structure("check", pdb_path)
    if chain_id in structure[0]:
        return chain_id, False

    atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"])
    new_atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"], use_author_fields=False)
    chain_id_mapping = [(x,y) for x,y in zip(atom_array.chain_id, new_atom_array.chain_id) if y == chain_id]
    
    assert len(set([x[0] for x in chain_id_mapping])) == 1
    
    new_chain_id = chain_id_mapping[0][0]
    return new_chain_id, True

class BaseDataset(Dataset):

    NONE_RETURN_LOAD_STRUCTURE = {
        "pdb_id": None, 
        "chain_id": None,
        "residue_range": None,
        "pdb_chain": None,
    }

    def __init__(self, *args, **kwargs):
        """
        in kwargs:
            data_path: data storage directory prefix
            target_field: target label name
            split: "train", "valid", or "test"
            py_logger: python logger
            tokenizer: sequence tokenizer or structural tokenzier
            in_memory: False
        """
        self.data_path = kwargs["data_path"]
        self.target_field = kwargs["target_field"]
        self.truncation_length = kwargs["truncation_length"]
        self.filter_length = kwargs["filter_length"]
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]
        self.structure_pad_token_id = kwargs["tokenizer"].pad_token_id
        self.multi_label = kwargs["multi_label"]
        self.is_global_or_local = kwargs["is_global_or_local"]
        self.PDB_DATA_DIR = kwargs["pdb_data_dir"]
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.data_name = kwargs["data_name"]

        self.use_continuous = kwargs["use_continuous"]
        # `use_sequence`` for BaseDataset is always set to True to pass sequence
        # information to models, while `use_sequence` for the model itself is 
        # False by default to disable using sequence during tokenization
        self.use_sequence = True

        # try to load pre-processed data
        target_split_file = self.get_target_file_name()
        
        if os.path.exists(target_split_file):
            self.data = torch.load(target_split_file, weights_only=False)
            self.py_logger.info(f"Loading from processed file {target_split_file},"
                                f"structured data of {len(self.data)} entries.")
        else:
            self.py_logger.info(f"Cannot load from processed file {target_split_file} "
                                f"for structured data")
            if dist.is_initialized():
                assert dist.get_world_size() == 1
            # process data entries from raw data, different for every datasets
            self.process_data_from_scratch(*args, **kwargs)

            # preprocess index mappings before loading PDB structures, different for every datasets
            self.prepare_structure_loading()
                
            self.load_all_structures()

            self.sanity_check()
            # save to disk
            self.save_structured_data()
            
        # Dataset sharding will be done in LightningDataModule

        # assign tokenizer if haven't been assign in `process_data_from_scratch`
        if not hasattr(self, "tokenizer"):
            self.tokenizer = kwargs["tokenizer"]

        self.patch_due_to_protokens()

        self.patch_for_TAPE_homo()

    def patch_due_to_protokens(self,):
        """filter because ProTokens cannot proceed proteins longer than 1024
        """
        len_limit = 1024
        new_data = []
        if self.data_name == "ConformationalSwitchDataset":
            for i in range(len(self.data)):
                if (len(self.data[i]["prot1_pdb_chain"].sequence) <= len_limit 
                    and len(self.data[i]["prot2_pdb_chain"].sequence) <= len_limit):
                    new_data.append(self.data[i])
        else:
            for i in range(len(self.data)):
                if len(self.data[i]["pdb_chain"].sequence) <= len_limit:
                    new_data.append(self.data[i])
            
        if len(new_data) != len(self.data):
            self.data = new_data
            self.py_logger.info(f"reduce sequence lengths because of ProTokens from {len(self.data)} to {len(new_data)}")

    def patch_for_TAPE_homo(self,):
        """
        Filter proteins causing error in TAPE RH, which are indexed at 11220 (out of 12071) and 11958 (out of 12070)
        Error Example: 
            Bio.PDB.PDBExceptions.PDBConstructionException: Blank altlocs in duplicate residue SER (' ', 22, ' ') of chain 'A'
        Error Explanation: https://biopython.org/wiki/Reading_large_PDB_files
        """
        if self.data_name == "TapeRemoteHomologyDataset" and self.split == "train":
            skip_index = 11220
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
            skip_index = 11958
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
    
            self.py_logger.info(f"reduce sequence lengths for TAPE Homo to {len(self.data)}")
    
    def get_target_file_name(self,):
        assert NotImplementedError

    def save_structured_data(self, ):
        file = self.get_target_file_name()
        torch.save(self.data, file)
        self.py_logger.info(f"Save the processed, structured data to disk: {file}")
    
    def prepare_structure_loading(self):
        assert NotImplementedError

    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        input_ids, labels, seqs = tuple(zip(*batch))
        disc_flag, cont_flag = [], []
        for key in ALL_TOKENIZER_TYPE["discretized"]:
            disc_flag.append(isinstance(self.tokenizer, eval(key)))
        for key in ALL_TOKENIZER_TYPE["continuous"]:
            cont_flag.append(isinstance(self.tokenizer, eval(key)))
        disc_flag, cont_flag = any(disc_flag), any(cont_flag)
        
        if cont_flag:
            length_list = [len(x) for x in input_ids]
        
        input_ids = pad_structures(input_ids, 
                        constant_value=self.structure_pad_token_id, 
                        truncation_length=self.truncation_length)
        # input_ids:
        ## discretized verson: [B, L] for structural ids
        ## continuous version: [B, L, hidden_dim] (e.g., for tokenizer like ProteinMPNN)
        
        if disc_flag:
            input_mask = input_ids == self.structure_pad_token_id
        elif cont_flag:
            input_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
            for i in range(len(input_mask)):
                input_mask[i][length_list[i]:] = False
            input_mask = ~input_mask
        else:
            raise NotImplementedError  
        # input_mask: always [B, L]

        try:
            labels = torch.LongTensor(labels)
            # labels: [B,] for global labels
        except:
            labels = pad_structures(labels, constant_value=-100,
                        truncation_length=self.truncation_length,
                        pad_length=input_ids.shape[1])
            assert labels.shape == input_ids.shape[:2]
            # labels: always [B, L] for local labels
        
        for seq in seqs:
            assert max(seq) < 26
        seqs = pad_structures(seqs, constant_value=26, # 26 alphabets
                        truncation_length=self.truncation_length,
                        pad_length=input_ids.shape[1])
        assert seqs.shape == input_ids.shape[:2]

        return {
            "input_list": (input_ids, input_mask, seqs),
            "targets": labels
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_pdb_chain(self, pdb_id, chain_id):
        try:
            file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
            protein_chain = WrappedProteinChain.from_cif(file, 
                                                chain_id=chain_id, id=pdb_id)
        except:
            self.py_logger.info(f"Cannot retrieve from local cluster, pdb_id: {pdb_id}, chain_id: {chain_id}")
            return None
        return protein_chain
    
    def _get_init_cnt_stats(self):
        return {}
    
    def load_structure(self, idx, cnt_stats):
        """
        Arguments:
            idx: index for self.data list
            cnt_stats: a dict to calculate statistics for unsable data entries
        Return:
            {
                "pdb_id": pdb_id, 
                "chain_id": chain_id,
                "residue_range": residue_range,
                "pdb_chain": pdb_chain, 
                "local_label": local_label # optional
            }
            # residue_range default as [""] to indicate the whole protein; 
            # e.g., ["6-100"] to indicate PDB residue_index ranging from 6 to 100
        """
        assert NotImplementedError
        
    def load_all_structures(self, ):
        """For each pdb_id in self.data[], load its pdb structures by
        calling self.load_structure()
        """
        process_global_rank = 0
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
        self.py_logger.info(f"Loading total {len(self.data)} structures on "
                            f"device {process_global_rank}")
        
        cnt_stats = self._get_init_cnt_stats()
        if self.fast_dev_run:
            self.data = self.data[:16]
        for i in tqdm(range(len(self.data))):
            res = self.load_structure(i, cnt_stats)
            
            for k in res.keys():
                self.data[i][k] = res[k]
            assert "pdb_id" in res
            assert "chain_id" in res
            assert "residue_range" in res
            assert "pdb_chain" in res

        self.py_logger.info(f"Processing all structures results in count "
                            f"statistics: {cnt_stats}")
        
        bg_time = time.time()
        new_data = []
        for i in range(len(self.data)):
            if not self.data[i]["pdb_id"] is None:
                new_data.append(self.data[i])
        ed_time = time.time()
        print("Timing: ", (ed_time - bg_time))

        self.py_logger.info(f"After loading structure filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def sanity_check(self):
        """Filter according to length
        """

        new_data = []
        for item in self.data:
            pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
            selected_indices = self._get_selected_indices(pdb_chain.residue_index, residue_range)
            if len(selected_indices) == 0:
                continue
            # filter proteins that are too long
            if len(selected_indices) > self.filter_length:
                continue
            new_data.append(item)
        self.data = new_data

        self.py_logger.info(f"After sanity check for selected residues, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")

    def _get_selected_indices(self, residue_index, residue_range):
        """
        Arguments:
            residue_range: residue range with format like ["5-10", "20-300"] (default [""])
        """
        rr = residue_range
        if len(rr) == 1 and rr[0] == "":
            return np.arange(len(residue_index))
        
        left = [eval(sep.split("-")[0]) for sep in rr]
        right = [eval(sep.split("-")[1]) for sep in rr]
        rr_indices = [x for l, r in zip(left, right) for x in list(range(l, r+1))]

        selected_indices = []
        for i, ridx in enumerate(residue_index):
            if ridx in rr_indices:
                selected_indices.append(i)

        return selected_indices # a list
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        # specifically defined for ATLAS, PretrainPDB, CASP14 and CAMEO
        file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
        return file
    
    def _get_item_structural_tokens(self, index, skip_check=False):
        
        item = self.data[index]
        if not skip_check:
            if "token_ids" in item:
                if self.is_global_or_local == "local":
                    assert len(item["token_ids"]) == len(item[self.target_field])
                return item["token_ids"], item[self.target_field], item["real_seqs"]
    
        pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
        pdb_id, chain_id = item["pdb_id"], item["chain_id"]
        pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)
        
        if self.data_name == "AtlasDataset":
            chain_id = " "
        else:
            # convert chain_id if necessary because some chain_id needs to 
            # use use_author_field (specified in biotite).
            # except atlas, other datasets' pdb_path is independent of chain_id; 
            # and for atlas, there is no need to transform chain_id
            chain_id, is_changed = convert_chain_id(pdb_path, chain_id)
        assigned_labels = item[self.target_field]
        assert pdb_chain is not None
        
        if self.is_global_or_local == "local":
            assert len(residue_range) == 1 and residue_range[0] == ""
        
            if self.data_name in "ProteinShakeBindingSiteDataset":
                label_residue_index = item["residue_index"]
            elif self.data_name in ["BioLIP2FunctionDataset", 
                "InterProFunctionDataset", "ProteinGLUEEpitopeRegionDataset", 
                "AtlasDataset"]:
                # all local labels already aligned to pdb_chain.residue_index
                label_residue_index = pdb_chain.residue_index
            else:
                raise NotImplementedError
            
            assert len(assigned_labels) == len(label_residue_index)


        # encode protein structure into token_ids
        if isinstance(self.tokenizer, WrappedESM3Tokenizer):
            # chain_id conversion is already automatically dealt with 
            # WrappedProteinChain, and produced pdb_chain
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedFoldSeekTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProTokensTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProteinMPNNTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedMIFTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedAIDOTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedCheapS1D64Tokenizer):
            # CheapS1D64 is continuous tokenizer
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        else:
            raise NotImplementedError
        
        assert len(token_ids) == len(residue_index)
        # code compatability in case token_ids store continuous reprs
        token_ids = token_ids.detach()
        assert len(residue_index) == len(seqs)
        
        if self.is_global_or_local == "local":
            # align residue_index and label_residue_index, so that token_ids align with assigned_labels
            org_len = len(token_ids)
            align_indices_1 = [i for i, x in enumerate(label_residue_index) if x in residue_index]
            label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
            assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()

            align_indices_2 = [i for i, x in enumerate(residue_index) if x in label_residue_index]
            residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
            seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]

            try:
                assert (residue_index == np.array(label_residue_index)).all()
            except:
                # deal with repeated residue indices and achieve exact match with alignment
                idx_list = list(set(residue_index.tolist() + label_residue_index))
                
                alphabet = Alphabet(idx_list)
                sim_score = np.diag(np.ones(len(idx_list)))
                substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
                seq1 = GeneralSequence(alphabet, label_residue_index)
                seq2 = GeneralSequence(alphabet, residue_index.tolist())
                alignment = align_optimal(seq1, seq2, substitution_matrix)
                
                alignment = alignment[0].trace
                align_indices_1, align_indices_2 = [], []
                for i in range(len(alignment)):
                    if (alignment[i] != -1).all():
                        align_indices_1.append(alignment[i][0])
                        align_indices_2.append(alignment[i][1])

                label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
                assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()
                residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
                seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]


            if org_len - len(token_ids) != 0:
                print(">> residue reduced by : ", org_len - len(token_ids))

        # select according to residue range constraints for some global tasks
        selected_indices = self._get_selected_indices(residue_index, residue_range)
        assert len(selected_indices) != 0
        
        token_ids = token_ids[selected_indices]
        seqs = np.array(seqs)[selected_indices].tolist()
        if self.is_global_or_local == "local":
            assigned_labels = np.array(assigned_labels)[selected_indices].tolist()

        # cache the tokens
        self.data[index]["token_ids"] = token_ids.to("cpu").detach().clone()
        self.data[index][self.target_field] = assigned_labels
        self.data[index]["real_seqs"] = seqs
        if self.is_global_or_local == "local":
            assert len(token_ids) == len(assigned_labels)
        return token_ids, assigned_labels, seqs # torch.Tensor, List

    def __getitem__(self, index: int):
        return self._get_item_structural_tokens(index)

    def additional_label_filtering_for_TAPE_homo(self, tokenizer_name):

        if self.data_name == "TapeRemoteHomologyDataset":
            """
            The original TAPE dataset consists of 1195 labels.
            Filter label class that has less than 50 protein samples in the 
            training dataset, reducing from 1195 labels to 45 labels
            """

            labels_to_filter = set([
                22, 36, 47, 51, 73, 77, 78, 84, 88, 90, 126, 153, 176, 295, 
                0, 3, 21, 39, 45, 59, 70, 97, 179,
                26, 49, 60, 81, 95, 113, 124, 133, 143, 178,
                13, 14, 18, 42, 52, 56, 61, 91, 132, 135, 180, 246
            ])
            labels_mapping = {x: i for i, x in enumerate(sorted(list(labels_to_filter)))}

            assert self.target_field == "fold_label"
            new_data = []
            for x in self.data:
                if x[self.target_field] in labels_to_filter:
                    x[self.target_field] = labels_mapping[x[self.target_field]]
                    new_data.append(x)
            self.data = new_data

        if self.data_name == "TapeRemoteHomologyDataset" and tokenizer_name == "protokens":
            # filter 1ldt.cif
            new_data = []
            for i in range(len(self.data)):
                if self.data[i]["pdb_id"] != "1ldt":
                    new_data.append(self.data[i])
            self.data = new_data
        
    def additional_preprocessing_for_TAPE_homo(self, tokenizer_name):
        """
        Some proteins are skipped, because for all their residues, at least 
        one backbone coordinates are NaN
        """
        if ((tokenizer_name == "proteinmpnn" or tokenizer_name == "mif") 
            and self.data_name == "TapeRemoteHomologyDataset"):
            
            if self.split == "train":
                skip_index = set([793, 796, 894, 1119, 1200, 1303, 1315, 1686, 1966, 2359, 
                            2583, 3302, 4239, 4406, 4769, 4904, 7669, 9642, 9903, 9933, 
                            9937, 10517, 11832, 11836, 11958])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data

            if self.split == "valid":
                
                skip_index = set([499, 619, 630])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_family_holdout":
                skip_index = set([41, 828, 1131])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_superfamily_holdout":
                skip_index = set([97, 111, 115, 129, 350])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
    
    def cache_all_tokenized(self):
        """Precompute all tokenization results"""
        
        flag_list, name_list, type_list = [], [], []
        for tp in ALL_TOKENIZER_TYPE:
            for key in ALL_TOKENIZER_TYPE[tp]:
                flag_list.append(isinstance(self.tokenizer, eval(key)))
                name_list.append(key.replace("Wrapped", "").replace("Tokenizer", "").lower())
                type_list.append(tp)
        flag = any(flag_list)
        
        if flag:
            index = np.nonzero(flag_list)[0].item()
            tokenizer_name = name_list[index]
            if isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
                tokenizer_name += f"_{self.tokenizer.ckpt_name}"

            # use continous reprs
            continuous_flag = self.use_continuous
            if type_list[index] == "continuous":
                # continous flag only for discretized tokenizers (i.e., VQ-VAE-based PSTs)
                # set to False to avoid redundancy for continuous tokenizers
                continuous_flag = False
            continuous_flag = "" if not continuous_flag else "_continuous"

            # use sequence ids
            sequence_flag = "" if not self.use_sequence else "_sequence"

            # cache file to avoid redundant tokenizing for the same tokenizer 
            # when tuning hyper-parameters 
            cache_file_name = self.get_target_file_name() + f"_{tokenizer_name}_tokenized{continuous_flag}{sequence_flag}"
            if os.path.exists(cache_file_name):
                new_data = torch.load(cache_file_name, weights_only=False)
                self.data = new_data
                self.additional_label_filtering_for_TAPE_homo(tokenizer_name)
                self.py_logger.info(f"Loading cahced tokenized data from {cache_file_name}")
                return
            else:
                self.py_logger.info(f"Cannot load cahced tokenized data from {cache_file_name}, caching now")
        else:
            raise NotImplementedError
                
        
        self.additional_preprocessing_for_TAPE_homo(tokenizer_name)

        # pre-checking
        for index in tqdm(range(len(self))):
            try:
                self[index]
            except:
                self.py_logger.info(f"[Error]: Something wrong for index {index} "
                                    f"when using {tokenizer_name}\n[Warning]: if "
                                    f"you're using your own PST, you can skip wrongly "
                                    f"indexed samples for your PST. But please be aware that "
                                    f"other PST benchmakred by the authors all used these samples")
                raise IndexError
        if flag:
            torch.save(self.data, cache_file_name)

    def shard(self, shard_idx: int, num_shards: int):
        """Shard the dataset inplace by keeping the every `num_shards`"""
        self.py_logger.info(f"Loading shard {shard_idx} with world size {num_shards}")

        indices = range(len(self))[shard_idx::num_shards]
        self.data = [self.data[i] for i in indices]

        self.py_logger.info("Done sharded loading.")

    
    def splitting_dataset(self, fold_split_ratio=0.4, fold_valid_ratio=0.2, 
        superfamily_split_ratio=0.2, superfamily_valid_ratio=0.2, seed=42
    ):
        """
        Perform splitting:
        - step 1: for each fold, split superfamilies into two groups (60%, 40%) 
            for training and test, resulting in the fold-level datasets
        - step 2: among the fold-level training data, for each superfamily, 
            split the family into two groups (60%, 40%) for training and test, 
            resulting in the superfamily-level datasets
        - Step 3: from the test data above, randomly take out 20% proteins 
            to create a validation set
        """

        # for each fold, split superfamilies
        fold_list, superfamily_list = [], []
        for i in range(len(self.data)):
            fold_list.append(self.data[i]["fold_label"])
            superfamily_list.append(self.data[i]["superfamily_label"])
        fold_list, superfamily_list = np.array(fold_list), np.array(superfamily_list)

        fold_train_indices, fold_test_indices = [], []
        for fold_idx in set(fold_list):
            indices = (fold_list == fold_idx).nonzero()[0]
            superfamily_vocab = list(set(superfamily_list[indices]))
            if int(len(superfamily_vocab) * fold_split_ratio) > 0:
                sf_train, sf_test = train_test_split(superfamily_vocab, 
                                        test_size=fold_split_ratio, random_state=seed)
                sf_train = np.isin(superfamily_list[indices], sf_train)
                sf_test = np.isin(superfamily_list[indices], sf_test)
                fold_train_indices += (indices[sf_train]).tolist()
                fold_test_indices += (indices[sf_test]).tolist()
            else:
                fold_train_indices += indices.tolist()

        fold_test_indices, fold_valid_indices = train_test_split(fold_test_indices, 
                                    test_size=fold_valid_ratio, random_state=seed)

        # among the fold-level training data, for each superfamily, random split 
        fold_train_indices = np.array(fold_train_indices)
        sf_train_indices, sf_test_indices = [], []
        for sf_idx in set(superfamily_list[fold_train_indices].tolist()):
            indices = (superfamily_list[fold_train_indices] == sf_idx).nonzero()[0]
            if int(len(indices) * superfamily_split_ratio) > 0:
                train_indices, test_indices = train_test_split(indices, 
                                        test_size=superfamily_split_ratio, random_state=seed)
                sf_train_indices += fold_train_indices[train_indices].tolist()
                sf_test_indices += fold_train_indices[test_indices].tolist()
            else:
                sf_train_indices += fold_train_indices[indices].tolist()

        sf_test_indices, sf_valid_indices = train_test_split(sf_test_indices, 
                                    test_size=superfamily_valid_ratio, random_state=seed)
        
        train_indices = sf_train_indices
        valid_indices = fold_valid_indices + sf_valid_indices
        fold_test_indices = fold_test_indices
        superfamily_test_indices = sf_test_indices

        assert len(train_indices) == len(set(train_indices))
        assert len(valid_indices) == len(set(valid_indices))
        assert len(fold_test_indices) == len(set(fold_test_indices))
        assert len(superfamily_test_indices) == len(set(superfamily_test_indices))
        assert len(self.data) == (len(set(train_indices)) + len(set(valid_indices))
                            + len(set(fold_test_indices)) + len(set(superfamily_test_indices)))

        self.py_logger.info(f"After splitting, result in {len(train_indices)} training data, "
                            f"{len(valid_indices)} validation data, {len(fold_test_indices)} fold-level test data, "
                            f"{len(superfamily_test_indices)} superfamily-level test data")
        
        train_data = [self.data[idx] for idx in train_indices]
        valid_data = [self.data[idx] for idx in valid_indices]
        fold_test_data = [self.data[idx] for idx in fold_test_indices]
        superfamily_test_data = [self.data[idx] for idx in superfamily_test_indices]
        return train_data, valid_data, fold_test_data, superfamily_test_data

    



def compute_embedding(item):
    _id, chain, _, _ = item
    protein = ESMProtein.from_protein_chain(ProteinChain.from_rcsb(_id, chain_id=chain))    
    # Ensure only one thread performs GPU operations at a time.
    with gpu_lock:
        protein_tensor = client.encode(protein)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
    embed = output.embeddings[0, 1:-1].to(torch.float32).to('cpu')
    return embed


def compute_embedding_from_pdb(item):
    _id, chain, _, _, prot_fname = item
    protein = ESMProtein.from_protein_chain(ProteinChain.from_pdb(prot_fname, chain_id=chain, is_predicted=True))    
    # Ensure only one thread performs GPU operations at a time.
    with gpu_lock:
        protein_tensor = client.encode(protein)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
    embed = output.embeddings[0, 1:-1].to(torch.float32).to('cpu')
    return embed


def traverse(tree):
    def _postorder(n, nodes, edges):
        assert (n.left is None) == (n.right is None), f"{n} {n.left} {n.right}"        
        if n.left:           
            level = _postorder(n.left, nodes, edges)
            level = max(level, _postorder(n.right, nodes, edges))            
            if level >= len(edges):
                edges.append([])
            edges[level].append((n, n.left, n.right))
            level += 1
        else: # reached leaf
            level = 0
        if level >= len(nodes):
            nodes.append([])
        nodes[level].append(n)
        return level
    all_edges = []
    all_nodes = []
    nmap = {}
    roots = [tree.nodes[k] for k in sorted(tree.nodes)]    
    for root in roots:
        nodes, edges = [], []
        _postorder(root, nodes, edges) # appends [[nodes of level i] for i]
        all_edges.append(edges)
        all_nodes.append(nodes[::-1])
    while True:
        stop = True
        for nodes in all_nodes:
            if len(nodes) == 0:
                continue
            stop = False
            for n in nodes.pop(-1):
                if n.value[-1] == 1:
                    assert n.value[0] == len(nmap)
                nmap[n.value] = len(nmap)
        if stop:
            break
    all_edges = [(nmap[p.value], nmap[l.value], nmap[r.value]) for edges in all_edges for edges_level in edges for p, l, r in edges_level]
    return nmap, all_edges


class MyDataset(Dataset):
    def __init__(self, tokenizers, dataset, label_map, debug=False):
        self.debug = debug
        mapping = {}
        for i, t in enumerate(tokenizers):
            stem = Path(t.fname).stem
            mapping[stem] = i
        my_data = []
        for sample in dataset:
            if self.debug and len(my_data) == 10:
                break
            prot, chain = sample['pdb_id'], sample['chain_id']
            key = f"{prot}_{chain}"
            breakpoint()
            if key in mapping:
                i = mapping[key]
                sample['fold_label'] = sample['fold_label']
                my_data.append((prot, chain, tokenizers[i], sample))
        self.data = my_data
        self.precompute()
    

    def precompute(self):
        # debug, comment out
        # self.esm_outputs = [torch.rand((sample['protein_length'], 960)).to(torch.float32).to('cpu') for _,_,_,sample in self.data]
        # return
        # end debug        
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(compute_embedding, self.data),
                total=len(self.data),
                desc="precomputing esm embeddings"
            ))        
        self.esm_outputs = results
        for i in range(len(results)):
            if len(results[i]) != self.data[i][2].n:
                breakpoint()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, chain, t, sample = self.data[idx]        
        item = sample
        # item['protein'] = protein
        # item['coords'] = t.compute_coords()
        tree = t.bond_to_token.tree
        nmap, edges = traverse(tree)
        item['edges'] = edges        
        item['embeddings'] = self.esm_outputs[idx]
        return item


class ResidueDataset(MyDataset):
    def __init__(self, tokenizers, dataset, debug=False):
        self.num_classes = 1 # binary classification
        self.debug = debug
        mapping = {}
        for i, t in enumerate(tokenizers):
            stem = Path(t.fname).stem
            mapping[stem] = i
        my_data = []
        assert len(dataset[0].keys()) == 3        
        poss_keys = [k for k in dataset[0].keys() if 'label' in k or 'score' in k]
        assert len(poss_keys) == 1
        label_key = poss_keys[0]
        for sample in dataset:
            if self.debug and len(my_data) == 10:
                break            
            prot, chain = sample['pdb_id'], sample['chain_id']
            key = f"{prot}_{chain}"            
            if key in mapping:
                i = mapping[key]
                if tokenizers[i].n != len(sample[label_key]):
                    breakpoint()
                    continue
                sample['residue_label'] = sample[label_key]                
                if len(ProteinChain.from_rcsb(prot, chain_id=chain)) == tokenizers[i].n:
                    my_data.append((prot, chain, tokenizers[i], sample))
                else:
                    breakpoint()
        print(f"{len(my_data)}/{len(dataset)} processed")
        self.data = my_data
        self.precompute()
