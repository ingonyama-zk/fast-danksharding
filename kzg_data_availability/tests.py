import os
import blst
import itertools
from time import time
import numpy as np
from fft import fft, expand_root_of_unity
import random
from typing import Tuple, List
from tqdm import tqdm
from kzg_proofs import (
    MODULUS,
    check_proof_multi,
    generate_setup,
    commit_to_poly,
    list_to_reverse_bit_order,
    get_root_of_unity,
    reverse_bit_order,
    is_power_of_two,
    eval_poly_at,
    get_extended_data,
    P1_INF
)

from fk20_multi import data_availabilty_using_fk20_multi

import csv
# 8x reduced size for testing, for full set use n = 512 and m = 4096
n = 32*8
m = 512*8
PRIMITIVE_ROOT = 5
MAX_DEGREE_POLY = MODULUS-1

test_vectors_path = f'../test_vectors/{n}x{m}/'
if not os.path.exists(test_vectors_path):
    os.makedirs(test_vectors_path)
    
if __name__ == "__main__":
    print('hi')

    # random input
    D_in = [[random.randint(1, MAX_DEGREE_POLY) for _ in range(m)] for _ in range(n)]
    
    # saved input
    # with open(test_vectors_path+'D_in.csv') as file:
    #     reader = csv.reader(file)
    #     D_in = [[int(i, 16) for i in row] for row in reader]

    # Coefficient form
    C_rows = [fft(D_in[i], MODULUS, get_root_of_unity(m), inv=True)
              for i in range(n)]
    print(hex(get_root_of_unity(m)))

    # Right branch
    seed = 32093670629883792236525422890014559721299643013210434231272819980374078291731
    # seed = random.getrandbits(256)
    setup = generate_setup(seed, m)
    SRS = setup[0]

    rootz_m = expand_root_of_unity(get_root_of_unity(2*m), MODULUS)[:-1]
    rootz_n = expand_root_of_unity(get_root_of_unity(2*n), MODULUS)[:-1]

    K0 = [commit_to_poly(C_rows[i], setup) for i in range(n)]

    B0 = fft(K0, MODULUS, get_root_of_unity(n), inv=True)
    B1 = [B0[i].dup().mult(rootz_n[i]) for i in range(n)]
    K1 = fft(B1, MODULUS, get_root_of_unity(n), inv=False)

    K = K0 + K1

    # Mid branch
    C_rows_t = np.array(C_rows).T.tolist()
    C_t = [fft(C_rows_t[i], MODULUS, get_root_of_unity(n), inv=True)
           for i in range(m)]
    C = np.array(C_t).T.tolist()

    C2 = [[(C[i][j]*rootz_n[i]) % MODULUS for j in range(m)] for i in range(n)]
    C0 = [[(C[i][j]*rootz_m[j]) % MODULUS for j in range(m)] for i in range(n)]

    E2 = [fft(C2[i], MODULUS, get_root_of_unity(m), inv=False)
          for i in range(n)]
    E0 = [fft(C0[i], MODULUS, get_root_of_unity(m), inv=False)
          for i in range(n)]

    E1 = [[(E0[i][j]*rootz_n[i]) % MODULUS for j in range(m)]
          for i in range(n)]

    E0_t = np.array(E0).T.tolist()
    D0_t = [fft(E0_t[i], MODULUS, get_root_of_unity(n), inv=False)
            for i in range(m)]
    D_rows = np.array(D0_t).T.tolist()

    E1_t = np.array(E1).T.tolist()
    D1_t = [fft(E1_t[i], MODULUS, get_root_of_unity(n), inv=False)
            for i in range(m)]
    D_both = np.array(D1_t).T.tolist()

    E2_t = np.array(E2).T.tolist()
    D2_t = [fft(E2_t[i], MODULUS, get_root_of_unity(n), inv=False)
            for i in range(m)]
    D_cols = np.array(D2_t).T.tolist()

    list_1 = D_in+D_cols
    list_2 = D_rows+D_both
    D_b4rbo = [[x for sub in list(zip(list_1[i], list_2[i]))
                for x in sub] for i in range(2*n)]

    D = [list_to_reverse_bit_order(D_b4rbo[i]) for i in range(2*n)]

    # Left branch
    l = 16
    invl = pow(l, MODULUS-2, MODULUS)
    S = fft([P1_INF.dup()] + SRS[-2-l::-1] + [P1_INF.dup()] *
            (m+l-1), MODULUS, get_root_of_unity(2*m), inv=False)
    [S[i].mult(invl) for i in range(2*m)]

    # First method
    t0 = time()
    d0 = [[S[j].dup().mult(D_b4rbo[i][j]) for j in range(2*m)]
          for i in range(2*n)]

    d1 = [[P1_INF.dup() for _ in range(2*m//l)] for _ in range(2*n)]
    for i in range(2*n):
        for j in range(2*m//l):
            for k in range(l):
                d1[i][j].add(d0[i][j+2*m//l*k])

    delta0 = [fft(d1[i], MODULUS, get_root_of_unity(2*m//l), inv=True)
              for i in range(2*n)]

    delta1 = [delta0[i][m//l:]+[P1_INF]*(m//l) for i in range(2*n)]

    q = [fft(delta1[i], MODULUS, get_root_of_unity(2*m//l), inv=False)
         for i in range(2*n)]

    P = [list_to_reverse_bit_order(q[i]) for i in range(2*n)]

    t1 = time()

    # Second method
    dd0 = [[S[j].dup().mult(D_b4rbo[i][j]) for j in range(2*m)]
           for i in range(n)]

    dd1 = [[P1_INF.dup() for _ in range(2*m//l)] for _ in range(n)]
    for i in range(n):
        for j in range(2*m//l):
            for k in range(l):
                dd1[i][j].add(dd0[i][j+2*m//l*k])

    deltad0 = [fft(dd1[i], MODULUS, get_root_of_unity(2*m//l), inv=True)
               for i in range(n)]

    deltad1 = [deltad0[i][m//l:]+[P1_INF]*(m//l) for i in range(n)]

    qd = [fft(deltad1[i], MODULUS, get_root_of_unity(2*m//l), inv=False)
          for i in range(n)]

    P0 = [list_to_reverse_bit_order(qd[i]) for i in range(n)]

    P0_t = np.array(P0).T.tolist()
    Q0 = [fft(P0_t[i], MODULUS, get_root_of_unity(n), inv=True)
          for i in range(2*m//l)]
    Q1 = [[Q0[i][j].dup().mult(rootz_n[j]) for j in range(n)]
          for i in range(2*m//l)]
    P1_t = [fft(Q1[i], MODULUS, get_root_of_unity(n), inv=False)
            for i in range(2*m//l)]
    P1 = np.array(P1_t).T.tolist()

    Pd = P0 + P1

    t2 = time()
    int1 = t1 - t0
    int2 = t2 - t1

    print("First method: %f seconds.\n" % int1)
    print("Second method: %f seconds.\n" % int2)

    # Ethereum method
    Se = []
    rou = get_root_of_unity(m//l * 2)
    for j in range(l):
        x = SRS[m - l - 1 - j::-l] + [P1_INF.dup()]*(m//l+1)
        Se.append(fft(x, MODULUS, rou, inv=False))

    t3 = time()
    de1 = [[P1_INF.dup()] * 2 * (m//l)]*n
    deltae0 = [[P1_INF.dup()] * 2 * (m//l)]*n
    qe = [[P1_INF.dup()] * 2 * (m//l)]*n
    for i in range(n):
        for j in range(l):
            C_resh = C_rows[i][- j - 1::l] + [0] * \
                (m//l + 1) + C_rows[i][2 * l - j - 1: - l - j:l]
            De0 = fft(C_resh, MODULUS, rou, inv=False)
            De1 = [v.dup().mult(w) for v, w in zip(Se[j], De0)]
            de1[i] = [v.dup().add(w) for v, w in zip(de1[i], De1)]
        deltae0[i] = fft(de1[i], MODULUS, rou, inv=True)[
            :m // l] + [P1_INF.dup()] * (m//l)
        qe[i] = fft(deltae0[i], MODULUS, rou, inv=False)

    Pe0 = [list_to_reverse_bit_order(qe[i]) for i in range(n)]

    Pe0_t = np.array(Pe0).T.tolist()
    Qe0 = [fft(Pe0_t[i], MODULUS, get_root_of_unity(n), inv=True)
           for i in range(2*m//l)]
    Qe1 = [[Qe0[i][j].dup().mult(rootz_n[j]) for j in range(n)]
           for i in range(2*m//l)]
    Pe1_t = [fft(Qe1[i], MODULUS, get_root_of_unity(n), inv=False)
             for i in range(2*m//l)]
    Pe1 = np.array(Pe1_t).T.tolist()

    Pe = Pe0 + Pe1

    t4 = time()
    int3 = t4 - t3
    print("Ethereum method: %f seconds.\n" % int3)

    # Check proofs

    rou = get_root_of_unity(m * 2)
    print("Testing...")
    for line in tqdm(range(2*n)):
        if line < n:
            polynomial = C_rows[line]
            commitment = commit_to_poly(polynomial, setup)
            assert K[line].serialize().hex() == commitment.serialize().hex()
            # print('Commitment ok')
            extended_data = get_extended_data(polynomial)
            assert all([D[line][i] == extended_data[i] for i in range(2*m)])
            # print('Extended data ok')
            all_proofs = data_availabilty_using_fk20_multi(
                polynomial, l, setup)

        for pos in range(2 * m // l):
            x = pow(rou, reverse_bit_order(pos, 2 * m // l), MODULUS)
            ys = D[line][l * pos:l * (pos + 1)]
            assert check_proof_multi(
                K[line], Pd[line][pos], x, list_to_reverse_bit_order(ys), setup)
        # print("Data availability line check {0} passed".format(line))

    print("All tests passed!")

    # Print to csv files

    # uncomment to write inputs
    with open(test_vectors_path+'D_in.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_in[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'C_rows.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C_rows[i][j]) for j in range(m)])
         for i in range(n)]

    with open(test_vectors_path+'K0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K0[i].serialize().hex()]) for i in range(n)]

    with open(test_vectors_path+'SRS.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([SRS[i].serialize().hex()]) for i in range(m)]

    with open(test_vectors_path+'roots_w.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(rootz_m[i])]) for i in range(m)]

    with open(test_vectors_path+'roots_u.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(rootz_n[i])]) for i in range(n)]

    with open(test_vectors_path+'B0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([B0[i].serialize().hex()]) for i in range(n)]

    with open(test_vectors_path+'B1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([B1[i].serialize().hex()]) for i in range(n)]

    with open(test_vectors_path+'K1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K1[i].serialize().hex()]) for i in range(n)]

    with open(test_vectors_path+'K.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([K[i].serialize().hex()]) for i in range(2*n)]

    with open(test_vectors_path+'C.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'D_rows.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_rows[i][j]) for j in range(m)])
         for i in range(n)]

    with open(test_vectors_path+'D_cols.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_cols[i][j]) for j in range(m)])
         for i in range(n)]

    with open(test_vectors_path+'D_both.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_both[i][j]) for j in range(m)])
         for i in range(n)]

    with open(test_vectors_path+'C0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C0[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'C2.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(C2[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'E0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E0[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'E1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E1[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'E2.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(E2[i][j]) for j in range(m)]) for i in range(n)]

    with open(test_vectors_path+'D_b4rbo.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D_b4rbo[i][j]) for j in range(2*m)])
         for i in range(2*n)]

    with open(test_vectors_path+'D.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([hex(D[i][j]) for j in range(2*m)])
         for i in range(2*n)]

    with open(test_vectors_path+'S.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([S[i].serialize().hex()]) for i in range(2*m)]

    with open(test_vectors_path+'d0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([d0[i][j].serialize().hex()
                         for j in range(2*m)]) for i in range(2*n)]

    with open(test_vectors_path+'d1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([d1[i][j].serialize().hex()
                         for j in range(2*m//l)]) for i in range(2*n)]

    with open(test_vectors_path+'delta0.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([delta0[i][j].serialize().hex()
                         for j in range(2*m//l)]) for i in range(2*n)]

    with open(test_vectors_path+'delta1.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([delta1[i][j].serialize().hex()
                         for j in range(2*m//l)]) for i in range(2*n)]

    with open(test_vectors_path+'q.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([q[i][j].serialize().hex()
                         for j in range(2*m//l)]) for i in range(2*n)]

    with open(test_vectors_path+'P.csv', 'w') as file:
        writer = csv.writer(file)
        [writer.writerow([P[i][j].serialize().hex()
                         for j in range(2*m//l)]) for i in range(2*n)]

    print('bye')
