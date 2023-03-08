use icicle_utils::{
    field::{PointAffineNoInfinity, Scalar, BASE_LIMBS, SCALAR_LIMBS},
    import_scalars, intt,
    matrix::{interleave_cols, rows_to_cols},
    msm, mult_sc_vec, ntt, store_random_points,
    utils::{from_limbs, import_limbs, list_to_reverse_bit_order},
    M_POINTS, N_ROWS,
};

pub fn get_debug_data(
    filename: &str,
    limbsize: usize,
    height: usize,
    lenght: usize,
) -> Vec<Vec<Scalar>> {
    let data_root_path = "../test_vectors/";

    let limbs = csv_to_u32_limbs(&format!("{}{}", data_root_path, filename), limbsize);

    let result = split_vec_to_matrix(&from_limbs(limbs, limbsize, Scalar::from_limbs), lenght);
    assert_eq!(result.len(), height);
    assert_eq!(result[0].len(), lenght);

    result
}

pub fn import_points(group_s_filepath: &str, expected_count: usize) -> Vec<PointAffineNoInfinity> {
    store_random_points(Some(2), expected_count, group_s_filepath); //TODO: disable

    let limbs = import_limbs(group_s_filepath);

    let points = from_limbs(limbs, 2 * BASE_LIMBS, PointAffineNoInfinity::from_xy_limbs);

    assert_eq!(points.len(), expected_count);
    points
}

#[allow(non_snake_case)]
pub fn main_flow() {
    println!(
        "new cuda output: {:?}",
        msm(&[PointAffineNoInfinity::default()], &[Scalar::default()], 0)
    );

    // let D_in: Vec<Vec<Scalar>> = (0..K_ROWS)
    //     .map(|_| generate_scalars(N_POINTS, get_rng(Some(3))))
    //     .collect();
    let D_in = get_debug_data("D_in.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);

    //TODO: all branches and many steps can be parallel, review after correctness
    ///////////////////
    let mut C_rows = D_in.clone();
    //C_rows = INTT_rows(D_in) //K_ROWS * N_POINTS (originally 256x4096)
    C_rows.iter_mut().for_each(|row| intt(row, 0));

    let C_rows_debug = get_debug_data("C_rows.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(C_rows, C_rows_debug);

    let tf_u = import_scalars("tf_u.bin", M_POINTS);

    ///////////////////

    ////////////////////////////////
    ///// branch 1
    ////////////////////////////////
    /*
        let s = import_points("s_.bin", SRS_SIZE);

        // K0 = MSM_rows(C_rows) (originally 256x1)
        //TODO: matrix of pointers so it can be processed without "rotation" of vectors
        let K0 = C_rows
            .clone() //TODO: clone not necessary?
            .iter()
            .map(|scalars| msm(&s, &scalars, 0usize))
            .collect::<Vec<_>>();

        assert!(K0.len() == K_ROWS);
        ////////////////////////////////

        // B0 = ECINTT_col(K0) N_POINTS x 1 (originally 256x1)
        //TODO: can be simplified, now follows diagram
        let mut B0 = K0.clone();
        iecntt(&mut B0, 0);

        // B1 = MUL_col(B0, [1 u u^2 ...]) N_POINTS x 1 (originally 256x1)
        let mut B1 = B0;
        multp_vec(&mut B1, &tf_u); //TODO:

        // K1 = ECNTT_col(B1) N_POINTS x 1 (originally 256x1)
        let mut K1 = B1;
        ecntt(&mut K1, 0); //TODO:
        assert_eq!(K1.len(), N_POINTS);

        // K = [K0, K1]  // 2*N_POINTS x 1 (originally 512x1) commitments
        let mut K = [K0, K1].concat();
        assert_eq!(K.len(), 2 * N_POINTS);
    */

    ////////////////////////////////
    ///// branch 2
    ////////////////////////////////

    // C = INTT_cols(C_rows) 256x4096 col
    let C = rows_to_cols(
        &(0..M_POINTS)
            .map(|i| {
                let mut col = C_rows.iter().map(|row| row[i]).collect::<Vec<Scalar>>();
                intt(&mut col, 0);
                col
            })
            .collect::<Vec<Vec<_>>>(),
    );

    let C_debug = get_debug_data("C.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(C, C_debug);

    let tf_u = import_scalars("tf_u.bin", M_POINTS);
    let tf_u_debug =
        rows_to_cols(&get_debug_data("roots32.csv", SCALAR_LIMBS, N_ROWS, 1))[0].to_vec();

    let tf_u = tf_u_debug.to_vec(); //TODO:  remove
    assert_eq!(tf_u, tf_u_debug);

    let tf_w = import_scalars("tf_w.bin", M_POINTS);
    let roots512 = get_debug_data("roots512.csv", SCALAR_LIMBS, M_POINTS, 1);
    let tf_w_debug = rows_to_cols(&roots512)[0].to_vec();

    let tf_w = tf_w_debug.to_vec(); //TODO:  remove
    assert_eq!(tf_w, tf_w_debug);

    // C0 = MUL_cols(C, [1 w w^2 ...]) 256x4096
    // C0 = [[(C[i][j]*rootz_m[j]) % MODULUS  for j in range(m)] for i in range(n)]
    let mut C0 = C.clone();
    C0.iter_mut().for_each(|row| mult_sc_vec(row, &tf_w, 0));

    let C0_debug = get_debug_data("C0.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(C0, C0_debug);

    // C2 = MUL_rows(C, [1 u u^2 ...]) 256x4096
    // C2 = [[(C[i][j]*rootz_n[i]) % MODULUS  for j in range(m)] for i in range(n)]
    let mut C2 = rows_to_cols(&C);
    C2.iter_mut().for_each(|row| mult_sc_vec(row, &tf_u, 0));
    let C2 = rows_to_cols(&C2);

    let C2_debug = get_debug_data("C2.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(C2, C2_debug);

    //E0 = NTT_rows(C0) 256x4096
    let mut E0 = C0; //TODO: rows_to_cols(&C0)?
    E0.iter_mut().for_each(|row| ntt(row, 0));

    let E0_debug = get_debug_data("E0.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(E0, E0_debug);

    //E1 = MUL_rows(E0, [1 u u^2 ...]) 256x4096
    let mut E1 = rows_to_cols(&E0);
    E1.iter_mut().for_each(|row| mult_sc_vec(row, &tf_u, 0));
    let E1 = rows_to_cols(&E1);

    let E1_debug = get_debug_data("E1.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(E1, E1_debug);

    // E2 = NTT_rows(C2) 256x4096
    let mut E2 = C2;
    E2.iter_mut().for_each(|row| ntt(row, 0));

    let E2_debug = get_debug_data("E2.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(E2, E2_debug);

    //D_rows = NTT_cols(E0) 256x4096
    let mut D_rows = rows_to_cols(&E0);
    D_rows.iter_mut().for_each(|row| ntt(row, 0));
    let D_rows = rows_to_cols(&D_rows);

    let D_rows_debug = get_debug_data("D_rows.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(D_rows, D_rows_debug);

    // D_both = NTT_cols(E1) 256x4096
    let mut D_both = rows_to_cols(&E1);
    D_both.iter_mut().for_each(|row| ntt(row, 0));
    let D_both = rows_to_cols(&D_both);

    let D_both_debug = get_debug_data("D_both.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(D_both, D_both_debug);

    // D_cols = NTT_cols(E2) 256x4096
    let mut D_cols = rows_to_cols(&E2);
    D_cols.iter_mut().for_each(|row| ntt(row, 0));
    let D_cols = rows_to_cols(&D_cols);

    let D_cols_debug = get_debug_data("D_cols.csv", SCALAR_LIMBS, N_ROWS, M_POINTS);
    assert_eq!(D_cols, D_cols_debug);

    //D_b4rbo = INTERLEAVE_cols( [D_in; D_cols], [D_rows; D_both] ) 512x8192
    let D_b4rbo = interleave_cols(&[D_in, D_cols].concat(), &[D_rows, D_both].concat());

    let D_b4rbo_debug = get_debug_data("D_b4rbo.csv", SCALAR_LIMBS, 2 * N_ROWS, 2 * M_POINTS);
    assert_eq!(D_b4rbo, D_b4rbo_debug);

    assert_eq!(D_b4rbo.len(), N_ROWS * 2);
    assert_eq!(D_b4rbo[0].len(), M_POINTS * 2);

    //D = RBO_cols(D_b4rbo) 512x8192
    //D = [list_to_reverse_bit_order(D_b4rbo[i]) for i in range(2*n)]
    let D = D_b4rbo
        .iter()
        .map(|row| list_to_reverse_bit_order(row))
        .collect::<Vec<_>>();

    let D_debug = get_debug_data("D.csv", SCALAR_LIMBS, 2 * N_ROWS, 2 * M_POINTS);
    assert_eq!(D, D_debug);

    ////////////////////////////////
    ///// branch 3
    ////////////////////////////////
}