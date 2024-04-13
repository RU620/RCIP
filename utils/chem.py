# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


def get_ecfp_from_smiles(
        smi: str, 
        radius: int=2, 
        dim: int=1024
        ) -> rdkit.DataStructs.cDataStructs.ExplicitBitVect:

    mol = Chem.MolFromSmiles(smi)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,dim)

    return ecfp

