"""Minimal stand-in for VaspBandUnfolding.unfold — only find_K_from_k is used by
wannier_unfolding.py. Transcribed verbatim from
VaspBandUnfolding/unfold.py (github.com/QijingZheng/VaspBandUnfolding) so behaviour is identical;
it avoids a dependency on the full package."""
import numpy as np

def find_K_from_k(k, M):
    M = np.array(M)
    Kc = np.dot(k, M.T)
    G = np.array(np.round(Kc), dtype=int)
    KG = Kc - np.round(Kc)
    return KG, G
