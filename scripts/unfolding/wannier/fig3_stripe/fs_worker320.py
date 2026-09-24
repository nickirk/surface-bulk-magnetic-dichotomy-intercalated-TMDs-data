"""Unfold the exact-320-band recipe-matched Hamiltonians.
Two grids: nk=241 (like-for-like against the earlier maps) and nk=181 (the convention
plot_co_fermi_surface_side_by_side.py expects). E_F is the GROUND-STATE value."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Wannier directories (320-band stripe calculations) and ground-state SCF Fermi levels
MATS={"Co1_3NbS2":(os.environ.get("W90_DIR_CONBS2","Co1_3NbS2_stripe_wannier"),7.8760),
      "Co1_3TaS2":(os.environ.get("W90_DIR_COTAS2","Co1_3TaS2_stripe_wannier"),6.4456)}
JOBS=[(m,s) for m in MATS for s in (1,2)]
idx=int(os.environ.get("SLURM_PROCID","0")); mat,spin=JOBS[idx]
wdir,ef=MATS[mat]
os.environ["W90_DIR"]=wdir; os.environ["W90_EFERMI"]=str(ef)
import wannier_unfolding as wu
print(f"[{idx}] {mat} spin {spin}  E_F={ef} (ground state)",flush=True)
st=wu.parse_win(wu.WIN_FILE); cen=wu.parse_wf_centres(wu.WOUT_FILES[spin],152)
beta=wu.assign_wannier_to_prim(cen,st)
R,wR,H_R,nw=wu.parse_hr_dat(wu.HR_FILES[spin])
B_rec=2*np.pi*np.linalg.inv(st.A_prim).T
for NK,KR in ((241,1.4),(181,1.35)):
    kx=np.linspace(-KR,KR,NK); KX,KY=np.meshgrid(kx,kx,indexing="xy")
    kc=np.stack([KX.ravel(),KY.ravel(),np.zeros(KX.size)],axis=1)
    kf=kc@np.linalg.inv(B_rec); kf[:,2]=0.0
    t0=time.time(); E,P=wu.compute_spectral_weight(kf,H_R,R,wR,beta,batch=256)
    out=f"{os.path.dirname(os.path.abspath(__file__))}/fs320_{mat}_spin{spin}_nk{NK}_kz0.00.npz"
    np.savez_compressed(out,E=E,P=P,kx=kx,efermi=ef,mat=mat,spin=spin,
                        A_prim=st.A_prim,nk=NK,krange=KR,kz=0.0,nbands=320)
    print(f"[{idx}]   nk={NK} done in {time.time()-t0:.0f}s -> {os.path.basename(out)}",flush=True)
