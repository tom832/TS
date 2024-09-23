import os
import warnings
from abc import ABC, abstractmethod
import numpy as np

import useful_rdkit_utils as uru

# try:
#     from openeye import oechem
#     from openeye import oeomega
#     from openeye import oeshape
#     from openeye import oedocking
#     import joblib
# except ImportError:
#     # Since openeye is a commercial software package, just pass with a warning if not available
#     warnings.warn(f"Openeye packages not available in this environment; do not attempt to use ROCSEvaluator or "
#                   f"FredEvaluator")
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from sqlitedict import SqliteDict

from ugi_rxn_mapper import ugi_rxn_mapper

from chemprop.train import make_predictions, load_model
from chemprop.args import PredictArgs, get_checkpoint_paths
# args = PredictArgs()
# args.features_generator =  ["rdkit_2d","ifg_drugbank_2","ugi_qmdesc_atom"]
# args.number_of_molecules = 2
# args.gpu = 0
# # args.checkpoint_paths = [' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_1/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_8/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_6/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_0/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_4/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_9/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_7/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_3/model_0/model.pt', ' /home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_5/model_0/model.pt']
# args.checkpoint_paths = ['/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt']
# args.no_features_scaling = False
# args.preds_path = "./preds.csv"
# mpnn_model = load_model(args=args)

def modular_click_reverse_rxn(
    triazole_smi: str = "CC1=CN=NN1C",
    return_mol=False,
    kekulize=True,
    canonical=True,
):
    """
    Generate reverse modular click (CuAAC + diazole transfer) product from modular click product SMILES string.

    Parameters
    ----------
    triazole_smi : str, default='C1=CC=NN1'
        SMILES string of modular click product triazole
    return_mol : bool, default=False
        If True, returns RDKit Mol object of reverse modular click product.
        If False, returns SMILES string of reverse modular click product
    kekulize : bool, default=True
        If True, returns kekulized SMILES string
    canonical : bool, default=True
        If True, returns canonical SMILES string

    Returns
    -------
    prod_mol : str or Chem.Mol
        If return_mol is False, returns SMILES string of reverse modular click product.
        If return_mol is True, returns RDKit Mol object of reverse modular click product.
    """
    triazole_mol = Chem.MolFromSmiles(triazole_smi)
    modular_click_reverse_smarts = "[#6:1][n:2]1:[n:3]:[n:4]:[c:6]:[c:5]1>>[NH2:2][#6:1].[N:3]=[N:4].[C:5]#[C:6]"
    modular_click_reverse_rxn = AllChem.ReactionFromSmarts(modular_click_reverse_smarts)
    products = modular_click_reverse_rxn.RunReactants((triazole_mol,))
    prod_amine_mol = products[0][0]
    prod_alkyne_mol = products[0][2]
    try:
        Chem.SanitizeMol(prod_amine_mol)
        Chem.SanitizeMol(prod_alkyne_mol)
    except:
        print(f"Error in sanitizing product for {triazole_smi}")
        return prod_amine_mol, prod_alkyne_mol
    try:
        if return_mol:
            return prod_amine_mol, prod_alkyne_mol
        else:
            return Chem.MolToSmiles(
                prod_amine_mol, kekuleSmiles=kekulize, canonical=canonical
            ), Chem.MolToSmiles(
                prod_alkyne_mol, kekuleSmiles=kekulize, canonical=canonical
            )
    except:
        print(f"Error in generating product for {triazole_smi}")
        return None
    

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, mol):
        pass

    @property
    @abstractmethod
    def counter(self):
        pass


class MWEvaluator(Evaluator):
    """A simple evaluation class that calculates molecular weight, this was just a development tool
    """

    def __init__(self):
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        return uru.MolWt(mol)


class FPEvaluator(Evaluator):
    """An evaluator class that calculates a fingerprint Tanimoto to a reference molecule
    """

    def __init__(self, input_dict):
        self.ref_smiles = input_dict["query_smiles"]
        self.ref_fp = uru.smi2morgan_fp(self.ref_smiles)
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, rd_mol_in):
        self.num_evaluations += 1
        rd_mol_fp = uru.mol2morgan_fp(rd_mol_in)
        return DataStructs.TanimotoSimilarity(self.ref_fp, rd_mol_fp)


# class ROCSEvaluator(Evaluator):
#     """An evaluator class that calculates a ROCS score to a reference molecule
#     """

#     def __init__(self, input_dict):
#         ref_filename = input_dict['query_molfile']
#         ref_fs = oechem.oemolistream(ref_filename)
#         self.ref_mol = oechem.OEMol()
#         oechem.OEReadMolecule(ref_fs, self.ref_mol)
#         self.max_confs = 50
#         self.score_cache = {}
#         self.num_evaluations = 0

#     @property
#     def counter(self):
#         return self.num_evaluations

#     def set_max_confs(self, max_confs):
#         """Set the maximum number of conformers generated by Omega
#         :param max_confs:
#         """
#         self.max_confs = max_confs

#     def evaluate(self, rd_mol_in):
#         """Generate conformers with Omega and evaluate the ROCS overlay of conformers to a reference molecule
#         :param rd_mol_in: Input RDKit molecule
#         :return: ROCS Tanimoto Combo score, returns -1 if conformer generation fails
#         """
#         self.num_evaluations += 1
#         smi = Chem.MolToSmiles(rd_mol_in)
#         # Look up to see if we already processed this molecule
#         arc_tc = self.score_cache.get(smi)
#         if arc_tc is not None:
#             tc = arc_tc
#         else:
#             fit_mol = oechem.OEMol()
#             oechem.OEParseSmiles(fit_mol, smi)
#             ret_code = generate_confs(fit_mol, self.max_confs)
#             if ret_code:
#                 tc = self.overlay(fit_mol)
#             else:
#                 tc = -1.0
#             self.score_cache[smi] = tc
#         return tc

#     def overlay(self, fit_mol):
#         """Use ROCS to overlay two molecules
#         :param fit_mol: OEMolecule
#         :return: Combo Tanimoto for the overlay
#         """
#         prep = oeshape.OEOverlapPrep()
#         prep.Prep(self.ref_mol)
#         overlay = oeshape.OEMultiRefOverlay()
#         overlay.SetupRef(self.ref_mol)
#         prep.Prep(fit_mol)
#         score = oeshape.OEBestOverlayScore()
#         overlay.BestOverlay(score, fit_mol, oeshape.OEHighestTanimoto())
#         return score.GetTanimotoCombo()


class LookupEvaluator(Evaluator):
    """A simple evaluation class that looks up values from a file.
    This is primarily used for testing.
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        ref_filename = input_dictionary['ref_filename']
        ref_df = pd.read_csv(ref_filename)
        self.ref_dict = dict([(a, b) for a, b in ref_df[['SMILES', 'val']].values])

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        return self.ref_dict[smi]

class DBEvaluator(Evaluator):
    """A simple evaluator class that looks up values from a database.
    This is primarily used for benchmarking
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        self.db_prefix = input_dictionary['db_prefix']
        db_filename = input_dictionary['db_filename']
        self.ref_dict = SqliteDict(db_filename)

    def __repr__(self):
        return "DBEvalutor"


    @property
    def counter(self):
        return self.num_evaluations


    def evaluate(self, smiles):
        self.num_evaluations += 1
        res = self.ref_dict.get(f"{self.db_prefix}{smiles}")
        if res is None:
            return np.nan
        else:
            if res == -500:
                return np.nan
            return res


# class FredEvaluator(Evaluator):
#     """An evaluator class that docks a molecule with the OEDocking Toolkit and returns the score
#     """

#     def __init__(self, input_dict):
#         du_file = input_dict["design_unit_file"]
#         if not os.path.isfile(du_file):
#             raise FileNotFoundError(f"{du_file} was not found or is a directory")
#         self.dock = read_design_unit(du_file)
#         self.num_evaluations = 0
#         self.max_confs = 50

#     @property
#     def counter(self):
#         return self.num_evaluations

#     def set_max_confs(self, max_confs):
#         """Set the maximum number of conformers generated by Omega
#         :param max_confs:
#         """
#         self.max_confs = max_confs

#     def evaluate(self, mol):
#         self.num_evaluations += 1
#         smi = Chem.MolToSmiles(mol)
#         mc_mol = oechem.OEMol()
#         oechem.OEParseSmiles(mc_mol, smi)
#         confs_ok = generate_confs(mc_mol, self.max_confs)
#         score = 1000.0
#         docked_mol = oechem.OEGraphMol()
#         if confs_ok:
#             ret_code = self.dock.DockMultiConformerMolecule(docked_mol, mc_mol)
#         else:
#             ret_code = oedocking.OEDockingReturnCode_ConformerGenError
#         if ret_code == oedocking.OEDockingReturnCode_Success:
#             dock_opts = oedocking.OEDockOptions()
#             sd_tag = oedocking.OEDockMethodGetName(dock_opts.GetScoreMethod())
#             # this is a stupid hack, I need to figure out how to do this correctly
#             oedocking.OESetSDScore(docked_mol, self.dock, sd_tag)
#             score = float(oechem.OEGetSDData(docked_mol, sd_tag))
#         return score


# def generate_confs(mol, max_confs):
#     """Generate conformers with Omega
#     :param max_confs: maximum number of conformers to generate
#     :param mol: input OEMolecule
#     :return: Boolean Omega return code indicating success of conformer generation
#     """
#     rms = 0.5
#     strict_stereo = False
#     omega = oeomega.OEOmega()
#     omega.SetRMSThreshold(rms)  # Word to the wise: skipping this step can lead to significantly different charges!
#     omega.SetStrictStereo(strict_stereo)
#     omega.SetMaxConfs(max_confs)
#     error_level = oechem.OEThrow.GetLevel()
#     # Turn off OEChem warnings
#     oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
#     status = omega(mol)
#     # Turn OEChem warnings back on
#     oechem.OEThrow.SetLevel(error_level)
#     return status


# def read_design_unit(filename):
#     """Read an OpenEye design unit
#     :param filename: design unit filename (.oedu)
#     :return: a docking grid
#     """
#     du = oechem.OEDesignUnit()
#     rfs = oechem.oeifstream()
#     if not rfs.open(filename):
#         oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

#     du = oechem.OEDesignUnit()
#     if not oechem.OEReadDesignUnit(rfs, du):
#         oechem.OEThrow.Fatal("Failed to read design unit")
#     if not du.HasReceptor():
#         oechem.OEThrow.Fatal("Design unit %s does not contain a receptor" % du.GetTitle())
#     dock_opts = oedocking.OEDockOptions()
#     dock = oedocking.OEDock(dock_opts)
#     dock.Initialize(du)
#     return dock


# def test_fred_eval():
#     """Test function for the Fred docking Evaluator
#     :return: None
#     """
#     input_dict = {"design_unit_file": "data/2zdt_receptor.oedu"}
#     fred_eval = FredEvaluator(input_dict)
#     smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
#     mol = Chem.MolFromSmiles(smi)
#     score = fred_eval.evaluate(mol)
#     print(score)


# def test_rocs_eval():
#     """Test function for the ROCS evaluator
#     :return: None
#     """
#     input_dict = {"query_molfile": "data/2chw_lig.sdf"}
#     rocs_eval = ROCSEvaluator(input_dict)
#     smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
#     mol = Chem.MolFromSmiles(smi)
#     combo_score = rocs_eval.evaluate(mol)
#     print(combo_score)


# class MLClassifierEvaluator(Evaluator):
#     """An evaluator class the calculates a score based on a trained ML model
#     """

#     def __init__(self, input_dict):
#         self.cls = joblib.load(input_dict["model_filename"])
#         self.num_evaluations = 0

#     @property
#     def counter(self):
#         return self.num_evaluations

#     def evaluate(self, mol):
#         self.num_evaluations += 1
#         fp = uru.mol2morgan_fp(mol)
#         return self.cls.predict_proba([fp])[:,1][0]


# def test_ml_classifier_eval():
#     """Test function for the ML Classifier Evaluator
#     :return: None
#     """
#     input_dict = {"model_filename": "mapk1_modl.pkl"}
#     ml_cls_eval = MLClassifierEvaluator(input_dict)
#     smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
#     mol = Chem.MolFromSmiles(smi)
#     score = ml_cls_eval.evaluate(mol)
#     print(score)


class UgiRxnMPNNEvaluator(Evaluator):
    """Ugi Rxn MPNN Evaluator class"""

    def __init__(self, input_dict):
        self.num_evaluations = 0
        self.args = PredictArgs()
        # self.args.features_generator =  ["rdkit_2d","ifg_drugbank_2","ugi_qmdesc_atom"]
        self.args.features_generator =  ["rdkit_2d","ifg_drugbank_2"]
        self.args.cal_qmdesc = True
        self.args.number_of_molecules = 2
        self.args.gpu = 0
        # self.args.checkpoint_paths = ['/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt']
        # self.args.checkpoint_paths = [
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_1/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_8/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_6/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_0/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_4/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_9/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_7/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_3/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_5/model_0/model.pt",
        # ]
        self.args.checkpoint_dir = "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test"
        self.args.checkpoint_paths = get_checkpoint_paths(checkpoint_dir=self.args.checkpoint_dir)
        # self.args.checkpoint_paths = [
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_2/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_1/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_8/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_6/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_0/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_4/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_9/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_7/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_3/model_0/model.pt",
        #     "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test/fold_5/model_0/model.pt",
        # ]
        self.args.no_features_scaling = False
        self.args.preds_path = "./preds.csv"
        self.mpnn_model = load_model(args=self.args)

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):

        self.num_evaluations += 1
        if isinstance(mol, str) == False:
            smi = Chem.MolToSmiles(mol)
        if smi is None:
            raise ValueError("Invaild Input Molecule")

        rxn_smi = ugi_rxn_mapper([smi])[0]
        rxn_smi = [[rxn_smi,"FC(F)(F)CO"]]

        # args = PredictArgs()
        # args.features_generator =  ["rdkit_2d","ifg_drugbank_2","ugi_qmdesc_atom"]
        # args.number_of_molecules = 2
        # args.gpu = 0
        # args.checkpoint_paths = ['/home/jnliu/chemprop/benchmark_chemprop/hyper_opt/opt_for_pred/trial_seed_60/fold_2/model_0/model.pt']
        # args.no_features_scaling = False
        # args.preds_path = "./preds.csv"
        # preds_result = make_predictions(args, rxn_smi)

        try:
            preds_result = make_predictions(self.args, smiles=rxn_smi, model_objects=self.mpnn_model)
        except:
            self.mpnn_model = load_model(args=self.args)
            preds_result = make_predictions(self.args, smiles=rxn_smi, model_objects=self.mpnn_model)

        return preds_result[0][0]


class ActivityMPNNEvaluator(Evaluator):
    """Activity MPNN Evaluator class"""

    def __init__(self, input_dict):
        self.num_evaluations = 0
        self.args = PredictArgs()
        # self.args.features_generator =  ["scaffoldkeys", "cats2d", "ifp3_7en8"]
        self.args.features_generator =  ["scaffoldkeys", "cats2d"]
        self.args.gpu = 0
        # self.args.checkpoint_dir = "/home/zxhuang/modular_click/machine_learning/chemprop_project_rescreen/opted_results/doc_1_2_3/scaffoldkeys_cats2d_ifp3_7en8_qmdesc/scaffold_balanced_noTest"
        self.args.checkpoint_dir = "/home/zxhuang/modular_click/machine_learning/chemprop_project_rescreen/opted_results/doc_1_2_3_4/scaffoldkeys_cats2d_qmdesc/scaffold_balanced_noTest"
        self.args.checkpoint_paths = get_checkpoint_paths(checkpoint_dir=self.args.checkpoint_dir)
        self.args.no_features_scaling = False
        self.args.preds_path = "./preds.csv"
        self.args.cal_qmdesc = True
        self.args.num_workers = 8
        self.mpnn_model = load_model(args=self.args)

    @property
    def counter(self):
        return self.num_evaluations
    
    def evaluate(self, mol):
        self.num_evaluations += 1
        if isinstance(mol, str) == False:
            smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
        else:
            smi = mol
        print(smi)
        if smi is None:
            raise ValueError("Invaild Input Molecule")

        # raise ValueError("Not Implemented Yet")
        try:
            preds_result = make_predictions(self.args, smiles=[[smi]], model_objects=self.mpnn_model)
        except:
            self.mpnn_model = load_model(args=self.args)
            try:
                preds_result = make_predictions(self.args, smiles=[[smi]], model_objects=self.mpnn_model)
            except Exception as e:
                print(e)
                return 0.
        print(preds_result)

        return preds_result[0][0]


class UnitedMPNNEvaluator(Evaluator):
    """United MPNN Evaluator class for Ugi Reaction and Activity"""

    def __init__(self, input_dict):
        self.num_evaluations = 0
        self.args_ugi = PredictArgs()
        self.args_ugi.features_generator =  ["rdkit_2d","ifg_drugbank_2"]
        self.args_ugi.number_of_molecules = 2
        self.args_ugi.gpu = 0
        self.args_ugi.checkpoint_dir = "/home/jnliu/chemprop/benchmark_chemprop/yield_pred/all_data/rdkit_ifg2_qmatom_hyper_opted_no_test"
        self.args_ugi.checkpoint_paths = get_checkpoint_paths(checkpoint_dir=self.args_ugi.checkpoint_dir)
        self.args_ugi.no_features_scaling = False
        self.args_ugi.preds_path = "./preds.csv"
        self.args_ugi.cal_qmdesc = True
        self.args_ugi.num_workers = 8
        self.mpnn_model_ugi = load_model(args=self.args_ugi)

        self.args_activity = PredictArgs()
        self.args_activity.features_generator =  ["scaffoldkeys", "cats2d"]
        self.args_activity.gpu = 0
        self.args_activity.checkpoint_dir = "/home/zxhuang/modular_click/machine_learning/chemprop_project_rescreen/opted_results/doc_1_2_3_4/scaffoldkeys_cats2d_qmdesc/scaffold_balanced_noTest"
        self.args_activity.checkpoint_paths = get_checkpoint_paths(checkpoint_dir=self.args_activity.checkpoint_dir)
        self.args_activity.no_features_scaling = False
        self.args_activity.preds_path = "./preds.csv"
        self.args_activity.cal_qmdesc = True
        self.args_activity.num_workers = 8
        self.mpnn_model_activity = load_model(args=self.args_activity)

    @property
    def counter(self):
        return self.num_evaluations
    
    def evaluate(self, mol):
        self.num_evaluations += 1
        if isinstance(mol, str) == False:
            smi = Chem.MolToSmiles(mol)
        else:
            smi = mol
        print(smi)
        if smi is None:
            raise ValueError("Invaild Input Molecule")
        
        amine_smi, alkyne_smi = modular_click_reverse_rxn(triazole_smi=smi)

        rxn_smi = ugi_rxn_mapper([alkyne_smi])[0]
        rxn_smi = [[rxn_smi,"FC(F)(F)CO"]]
        try:
            preds_result_ugi = make_predictions(self.args_ugi, smiles=rxn_smi, model_objects=self.mpnn_model_ugi)
        except:
            self.mpnn_model_ugi = load_model(args=self.args_ugi)
            preds_result_ugi = make_predictions(self.args_ugi, smiles=rxn_smi, model_objects=self.mpnn_model_ugi)
        print('Ugi rxn feasibility:', preds_result_ugi)

        try:
            preds_result_activity = make_predictions(self.args_activity, smiles=[[smi]], model_objects=self.mpnn_model_activity)
        except:
            self.mpnn_model_activity = load_model(args=self.args_activity)
            preds_result_activity = make_predictions(self.args_activity, smiles=[[smi]], model_objects=self.mpnn_model_activity)
        print('Activity:', preds_result_activity)

        # Geometric Mean
        geometric_mean = np.sqrt(preds_result_ugi[0][0] * preds_result_activity[0][0])
        print('Geometric Mean:', geometric_mean)
        return geometric_mean


if __name__ == "__main__":
    # test_rocs_eval()
    pass
