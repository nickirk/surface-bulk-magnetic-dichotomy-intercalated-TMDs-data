#!/usr/bin/env python3
"""POSCAR -> VESTA-readable .vasp + .cif (P1).

VESTA opens a POSCAR only if it is named *.vasp, so the .vasp copy is the file
you actually double-click. The .cif is P1 with explicit fractional coordinates,
for anything that will not read VASP format.
"""
import sys, os, math

def read_poscar(p):
    L = open(p).read().split('\n')
    scale = float(L[1])
    lat = [[float(x)*scale for x in L[i].split()] for i in (2,3,4)]
    names = L[5].split(); counts = [int(x) for x in L[6].split()]
    mode = L[7].strip()[0].lower()
    n = sum(counts)
    pos = [[float(x) for x in L[8+i].split()[:3]] for i in range(n)]
    if mode == 'c':                                   # Cartesian -> fractional
        inv = invert(lat)
        pos = [[sum(q[k]*inv[k][j] for k in range(3)) for j in range(3)] for q in pos]
    return L[0].strip(), lat, names, counts, pos

def invert(m):
    d = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
        -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
        +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    c = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            a = [[m[r][cc] for cc in range(3) if cc != j] for r in range(3) if r != i]
            mi = a[0][0]*a[1][1] - a[0][1]*a[1][0]
            c[j][i] = ((-1)**(i+j)) * mi / d
    return c

def cellpar(lat):
    a, b, c = (math.sqrt(sum(v*v for v in r)) for r in lat)
    ang = lambda u, v: math.degrees(math.acos(sum(u[i]*v[i] for i in range(3))
                                    / (math.sqrt(sum(x*x for x in u))*math.sqrt(sum(x*x for x in v)))))
    return a, b, c, ang(lat[1],lat[2]), ang(lat[0],lat[2]), ang(lat[0],lat[1])

def write_cif(out, title, lat, names, counts, pos, magline=None):
    a,b,c,al,be,ga = cellpar(lat)
    L = ["# " + title]
    if magline: L.append("# " + magline)
    L += ["data_structure",
          f"_cell_length_a    {a:.6f}", f"_cell_length_b    {b:.6f}", f"_cell_length_c    {c:.6f}",
          f"_cell_angle_alpha {al:.4f}", f"_cell_angle_beta  {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
          "_symmetry_space_group_name_H-M   'P 1'", "_symmetry_Int_Tables_number      1",
          "loop_", " _symmetry_equiv_pos_as_xyz", " 'x, y, z'",
          "loop_", " _atom_site_label", " _atom_site_type_symbol",
          " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy"]
    k = 0
    for el, n in zip(names, counts):
        for i in range(n):
            x, y, z = (v - math.floor(v) for v in pos[k])
            L.append(f" {el+str(i+1):<7s} {el:<3s} {x:.6f} {y:.6f} {z:.6f} 1.0")
            k += 1
    open(out, 'w').write("\n".join(L) + "\n")

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
made = 0
for f in sorted(os.listdir(src_dir)):
    if not f.endswith('.POSCAR'): continue
    stem = f[:-7]
    p = os.path.join(src_dir, f)
    title, lat, names, counts, pos = read_poscar(p)
    magf = os.path.join(src_dir, stem + '.MAG')
    magline = open(magf).read().strip() if os.path.exists(magf) else None
    open(os.path.join(out_dir, stem + '.vasp'), 'w').write(open(p).read())
    write_cif(os.path.join(out_dir, stem + '.cif'), title, lat, names, counts, pos, magline)
    a,b,c,al,be,ga = cellpar(lat)
    print(f"{stem:32s} {sum(counts):4d} atoms  a={a:.3f} b={b:.3f} c={c:.3f}  gamma={ga:.2f}")
    made += 1
print(f"\n{made} structures -> .vasp + .cif in {out_dir}")
