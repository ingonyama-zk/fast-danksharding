import blst
import itertools
import numpy as np
from fft import fft, expand_root_of_unity
import random
from typing import Tuple, List
from tqdm import tqdm
from kzg_proofs import (
    MODULUS,
    check_proof_single,
    generate_setup,
    commit_to_poly,
    list_to_reverse_bit_order,
    get_root_of_unity,
    reverse_bit_order,
    is_power_of_two,
    eval_poly_at,
    P1_INF 
)
import csv
n = 32
m = 512
PRIMITIVE_ROOT = 5
MAX_DEGREE_POLY = MODULUS-1


if __name__ == "__main__":
    print('hi')
    random.seed(a=6, version=2)    

    ######## random input
    D_in = [[random.randint(1, MAX_DEGREE_POLY) for _ in range(m)] for _ in range(n)]

    ######## Coefficient form
    C_rows = [fft(D_in[i], MODULUS, get_root_of_unity(m), inv=True) for i in range(n)]
    # D_in1 = [fft(C_rows[i], MODULUS, get_root_of_unity(m), inv=False) for i in range(n)]

    ######## Right branch
    setup = generate_setup(random.getrandbits(256), m)
    SRS = setup[0]

    rootz_m = expand_root_of_unity(get_root_of_unity(m), MODULUS)[:-1]
    rootz_n = expand_root_of_unity(get_root_of_unity(n), MODULUS)[:-1]

    K0 = [commit_to_poly(C_rows[i],setup) for i in range(n)]
    # K0_t = np.array(K0).T.tolist()

    B0 = fft(K0,MODULUS,get_root_of_unity(n),inv=True)
    B1 = [B0[i].dup().mult(rootz_n[i]) for i in range(n)]
    K1 = fft(B1,MODULUS,get_root_of_unity(n),inv=False)

    K = K0 + K1

    ######## Mid branch
    C_rows_t = np.array(C_rows).T.tolist()
    C_t = [fft(C_rows_t[i], MODULUS, get_root_of_unity(n), inv=True) for i in range(m)]
    C = np.array(C_t).T.tolist()

    C2 = [[(C[i][j]*rootz_n[i]) % MODULUS  for j in range(m)] for i in range(n)]
    C0 = [[(C[i][j]*rootz_m[j]) % MODULUS  for j in range(m)] for i in range(n)]

    E2 = [fft(C2[i], MODULUS, get_root_of_unity(m), inv=False) for i in range(n)]
    E0 = [fft(C0[i], MODULUS, get_root_of_unity(m), inv=False) for i in range(n)]

    E1 = [[(E0[i][j]*rootz_n[i]) % MODULUS  for j in range(m)] for i in range(n)]

    E0_t = np.array(E0).T.tolist()
    D0_t = [fft(E0_t[i], MODULUS, get_root_of_unity(n), inv=False) for i in range(m)]
    D_rows = np.array(D0_t).T.tolist()

    E1_t = np.array(E1).T.tolist()
    D1_t = [fft(E1_t[i], MODULUS, get_root_of_unity(n), inv=False) for i in range(m)]
    D_both = np.array(D1_t).T.tolist()

    E2_t = np.array(E2).T.tolist()
    D2_t = [fft(E2_t[i], MODULUS, get_root_of_unity(n), inv=False) for i in range(m)]
    D_cols = np.array(D2_t).T.tolist()

    list_1 = D_in+D_cols
    list_2 = D_rows+D_both
    D_b4rbo = [[x for sub in list(zip(list_1[i],list_2[i])) for x in sub] for i in range(2*n)]

    D = [list_to_reverse_bit_order(D_b4rbo[i]) for i in range(2*n)]

    ######## Left branch

    S = fft(SRS[-2::-1]+[P1_INF]*m,MODULUS,get_root_of_unity(2*m),inv=False)

    d0 = [[S[j].dup().mult(D_b4rbo[i][j]) for j in range(2*m)] for i in range(2*n)]
    l = 16
    d1 = [[P1_INF]*(2*m//l)]*(2*n)
    for i in range(2*n):
        for j in range(2*m//l):
            for k in range(l):
                d1[i][j].dup().add(d0[i][j+2*m//l*k])
    # d1 = [[sum([d0[i][j+2*m//l*k] for k in range(l)]) for j in range(2*m//l)] for i in range(2*n)]

    d2 = [d1[i][m//l:]+[P1_INF]*(m//l) for i in range(2*n)]

    q = [fft(d2[i], MODULUS, get_root_of_unity(2*m//l), inv=True) for i in range(2*n)]

    P = [list_to_reverse_bit_order(q[i]) for i in range(2*n)]

    ######## Print to csv files

    with open('../wrapper/data/debug/D_in.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_in[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/C_rows.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C_rows[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/K0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K0[i].serialize().hex()]) for i in range(n)]

    with open('../wrapper/data/debug/SRS.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([SRS[i].serialize().hex()]) for i in range(m)]

    with open('../wrapper/data/debug/roots512.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(rootz_m[i])]) for i in range(m)]

    with open('../wrapper/data/debug/roots32.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(rootz_n[i])]) for i in range(n)]

    with open('../wrapper/data/debug/B0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([B0[i].serialize().hex()]) for i in range(n)]

    with open('../wrapper/data/debug/B1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([B1[i].serialize().hex()]) for i in range(n)]

    with open('../wrapper/data/debug/K1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K1[i].serialize().hex()]) for i in range(n)]

    with open('../wrapper/data/debug/K.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K[i].serialize().hex()]) for i in range(2*n)]

    with open('../wrapper/data/debug/C.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/D_rows.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_rows[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/D_cols.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_cols[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/D_both.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_both[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/C0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C0[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/C2.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C2[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/E0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E0[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/E1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E1[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/E2.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E2[i][j]) for j in range(m)]) for i in range(n)]

    with open('../wrapper/data/debug/D_b4rbo.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_b4rbo[i][j]) for j in range(2*m)]) for i in range(2*n)]

    with open('../wrapper/data/debug/D.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D[i][j]) for j in range(2*m)]) for i in range(2*n)]

    with open('../wrapper/data/debug/S.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([S[i].serialize().hex()]) for i in range(2*m)]

    with open('../wrapper/data/debug/d0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([d0[i][j].serialize().hex() for j in range(2*m)]) for i in range(2*n)]

    with open('../wrapper/data/debug/d1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([d1[i][j].serialize().hex() for j in range(2*m//l)]) for i in range(2*n)]

    with open('../wrapper/data/debug/d2.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([d2[i][j].serialize().hex() for j in range(2*m//l)]) for i in range(2*n)]

    with open('../wrapper/data/debug/q.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([q[i][j].serialize().hex() for j in range(2*m//l)]) for i in range(2*n)]

    with open('../wrapper/data/debug/P.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([P[i][j].serialize().hex() for j in range(2*m//l)]) for i in range(2*n)]

    print('bye')