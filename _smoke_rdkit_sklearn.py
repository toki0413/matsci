"""ponytail self-check: RDKit+sklearn GPR 在 subprocess 能跑通."""
import subprocess, sys
script = """
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
mol = Chem.MolFromSmiles('CCO')
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
arr = np.zeros((256,), dtype=np.float64)
ConvertToNumpyArray(fp, arr)
X = arr.reshape(1, -1)
y = np.array([50.0])
gpr = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1.0)
gpr.fit(X, y)
print("OK")
"""
result = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=60)
if result.returncode != 0:
    print(f"FAILED:\nstdout:{result.stdout}\nstderr:{result.stderr}")
    sys.exit(1)
assert "OK" in result.stdout
print("self-check passed: RDKit+sklearn GPR works in subprocess")
