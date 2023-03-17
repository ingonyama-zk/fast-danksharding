use std::time::Instant;

use icicle_utils::{
    field::{Point, Scalar},
    *,
};

use crate::{*, matrix::*, utils::*};

pub const FLOW_SIZE: usize = 1 << 12; //4096  //prod flow size
pub const TEST_SIZE_DIV: usize = 1; //TODO: Prod size / test size for speedup
pub const TEST_SIZE: usize = FLOW_SIZE / TEST_SIZE_DIV; //test flow size
pub const M_POINTS: usize = TEST_SIZE;
pub const SRS_SIZE: usize = M_POINTS;
pub const S_GROUP_SIZE: usize = 2 * M_POINTS;
pub const N_ROWS: usize = 256 / TEST_SIZE_DIV;
pub const FOLD_SIZE: usize = 512 / TEST_SIZE_DIV;

//TODO: the casing is to match diagram
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
pub fn main_flow() {
    let o = Instant::now();
    let D_in = get_debug_data_scalars("D_in.csv", N_ROWS, M_POINTS);

    //TODO: branches and many steps can be parallel
    ///////////////////
    let mut C_rows = D_in.clone();
    //C_rows = INTT_rows(D_in) //K_ROWS * N_POINTS (originally 256x4096)
    C_rows.iter_mut().for_each(|row| intt(row, 0));

    let C_rows_debug = get_debug_data_scalars("C_rows.csv", N_ROWS, M_POINTS);

    assert_eq!(C_rows, C_rows_debug);

    let tf_u = &get_debug_data_scalars("roots_u.csv", 1, N_ROWS)[0]; //TODO: csv

    println!("pre-branch {:0.3?}", o.elapsed());
    let o = Instant::now();

    ////////////////////////////////
    println!("Branch 1");
    ////////////////////////////////

    //[s] = SRS 1x4096
    let SRS = get_debug_data_points_proj_xy1_vec("SRS.csv", M_POINTS);

    debug_assert!(SRS[0].to_ark_affine().is_on_curve());

    let s_affine: Vec<_> = SRS.iter().map(|p| p.to_xy_strip_z()).collect();

    // K0 = MSM_rows(C_rows) (originally 256x1)
    let K0 = C_rows
        .clone() //TODO: clone not necessary?
        .iter()
        .map(|scalars| msm(&s_affine, &scalars, 0usize))
        .collect::<Vec<_>>();

    let K0_debug = get_debug_data_points_proj_xy1_vec("K0.csv", N_ROWS);
    debug_assert_eq!(K0, K0_debug);

    // B0 = ECINTT_col(K0) N_POINTS x 1 (originally 256x1)
    let mut B0 = K0.clone();
    iecntt(&mut B0, 0);

    let B0_debug = get_debug_data_points_proj_xy1_vec("B0.csv", N_ROWS);
    debug_assert_eq!(B0, B0_debug);

    // B1 = MUL_col(B0, [1 u u^2 ...]) N_POINTS x 1 (originally 256x1)
    let mut B1 = B0.clone();
    multp_vec(&mut B1, &tf_u, 0);

    let B1_debug = get_debug_data_points_proj_xy1_vec("B1.csv", N_ROWS);
    debug_assert_eq!(B1, B1_debug);

    // K1 = ECNTT_col(B1) N_POINTS x 1 (originally 256x1)
    let mut K1 = B1;
    ecntt(&mut K1, 0);
    let K1_debug = get_debug_data_points_proj_xy1_vec("K1.csv", N_ROWS);
    debug_assert_eq!(K1, K1_debug);

    // K = [K0, K1]  // 2*N_POINTS x 1 (originally 512x1) commitments
    let K = [K0, K1].concat();

    let K_debug = get_debug_data_points_proj_xy1_vec("K.csv", 2 * N_ROWS);
    debug_assert_eq!(K, K_debug);

    println!("Branch1 {:0.3?}", o.elapsed());
    let o = Instant::now();

    ////////////////////////////////
    println!("Branch 2");
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

    let C_debug = get_debug_data_scalars("C.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(C, C_debug);

    let roots_w = get_debug_data_scalars("roots_w.csv", M_POINTS, 1);
    let tf_w = rows_to_cols(&roots_w)[0].to_vec().to_vec();

    // C0 = MUL_cols(C, [1 w w^2 ...]) 256x4096
    let mut C0 = C.clone();
    C0.iter_mut().for_each(|row| mult_sc_vec(row, &tf_w, 0));

    let C0_debug = get_debug_data_scalars("C0.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(C0, C0_debug);

    // C2 = MUL_rows(C, [1 u u^2 ...]) 256x4096
    let mut C2 = rows_to_cols(&C);
    C2.iter_mut().for_each(|row| mult_sc_vec(row, &tf_u, 0));
    let C2 = rows_to_cols(&C2);

    let C2_debug = get_debug_data_scalars("C2.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(C2, C2_debug);

    //E0 = NTT_rows(C0) 256x4096
    let mut E0 = C0; //TODO: rows_to_cols(&C0)?
    E0.iter_mut().for_each(|row| ntt(row, 0));

    let E0_debug = get_debug_data_scalars("E0.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(E0, E0_debug);

    //E1 = MUL_rows(E0, [1 u u^2 ...]) 256x4096
    let mut E1 = rows_to_cols(&E0);
    E1.iter_mut().for_each(|row| mult_sc_vec(row, &tf_u, 0));
    let E1 = rows_to_cols(&E1);

    let E1_debug = get_debug_data_scalars("E1.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(E1, E1_debug);

    // E2 = NTT_rows(C2) 256x4096
    let mut E2 = C2;
    E2.iter_mut().for_each(|row| ntt(row, 0));

    let E2_debug = get_debug_data_scalars("E2.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(E2, E2_debug);

    //D_rows = NTT_cols(E0) 256x4096
    let mut D_rows = rows_to_cols(&E0);
    D_rows.iter_mut().for_each(|row| ntt(row, 0));
    let D_rows = rows_to_cols(&D_rows);

    let D_rows_debug = get_debug_data_scalars("D_rows.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(D_rows, D_rows_debug);

    // D_both = NTT_cols(E1) 256x4096
    let mut D_both = rows_to_cols(&E1);
    D_both.iter_mut().for_each(|row| ntt(row, 0));
    let D_both = rows_to_cols(&D_both);

    let D_both_debug = get_debug_data_scalars("D_both.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(D_both, D_both_debug);

    // D_cols = NTT_cols(E2) 256x4096
    let mut D_cols = rows_to_cols(&E2);
    D_cols.iter_mut().for_each(|row| ntt(row, 0));
    let D_cols = rows_to_cols(&D_cols);

    let D_cols_debug = get_debug_data_scalars("D_cols.csv", N_ROWS, M_POINTS);
    debug_assert_eq!(D_cols, D_cols_debug);

    //D_b4rbo = INTERLEAVE_cols( [D_in; D_cols], [D_rows; D_both] ) 512x8192
    let D_b4rbo = interleave_cols(&[D_in, D_cols].concat(), &[D_rows, D_both].concat());

    let D_b4rbo_debug = get_debug_data_scalars("D_b4rbo.csv", 2 * N_ROWS, 2 * M_POINTS);
    debug_assert_eq!(D_b4rbo, D_b4rbo_debug);

    debug_assert_eq!(D_b4rbo.len(), N_ROWS * 2);
    debug_assert_eq!(D_b4rbo[0].len(), M_POINTS * 2);

    //D = RBO_cols(D_b4rbo) 512x8192
    let D = D_b4rbo
        .iter()
        .map(|row| list_to_reverse_bit_order(row))
        .collect::<Vec<_>>();

    let D_debug = get_debug_data_scalars("D.csv", 2 * N_ROWS, 2 * M_POINTS);

    debug_assert_eq!(D, D_debug);

    println!("Branch2 {:0.3?}", o.elapsed());
    let o = Instant::now();

    ////////////////////////////////
    println!("Branch 3");
    ////////////////////////////////

    //TODO: now S is preprocessed, copy preprocessing here
    let S_debug = get_debug_data_points_proj_xy1_vec("S.csv", 2 * M_POINTS);

    let S = S_debug.clone();

    let d0_debug = get_debug_data_points_xy1("d0.csv", 2 * N_ROWS, 2 * M_POINTS);
    let d1_debug = get_debug_data_points_xy1("d1.csv", 2 * N_ROWS, 2 * N_ROWS);

    let mut q_ = Vec::<Vec<Point>>::new();

    let q_debug = get_debug_data_points_xy1("q.csv", 2 * N_ROWS, 2 * N_ROWS);

    let delta0_debug = get_debug_data_points_xy1("delta0.csv", 2 * N_ROWS, 2 * N_ROWS);

    let P_debug = get_debug_data_points_xy1("P.csv", 2 * N_ROWS, 2 * N_ROWS).clone();

    let delta1_debug = get_debug_data_points_xy1("delta1.csv", 2 * N_ROWS, 2 * N_ROWS);

    const l: usize = 16;

    println!("loaded test data, processing...");

    //d0 = MUL_row(d[mu], [S]) 1x8192
    let d0: Vec<_> = (0..2 * N_ROWS)
        .map(|i| {
            let mut s = S.clone();
            multp_vec(&mut s, &D_b4rbo[i], 0);
            s
        })
        .collect();
    debug_assert_eq!(d0, d0_debug);

    let mut d1_flat = vec![Point::infinity(); (2 * N_ROWS) * (2 * M_POINTS / l)];
    let d0_flat: Vec<_> = d0.into_iter().flatten().collect();

    addp_vec(&mut d1_flat, &d0_flat, 2 * N_ROWS, 2 * M_POINTS, l, 0);

    let d1 = split_vec_to_matrix(&d1_flat, 2 * N_ROWS).clone();
    debug_assert_eq!(d1, d1_debug);

    let mut delta0_flat: Vec<_> = d1.into_iter().flatten().collect();
    println!("iecntt batch for delta0");
    iecntt_batch(&mut delta0_flat, 2 * N_ROWS, 0);

    let mut delta0 = split_vec_to_matrix(&delta0_flat, 2 * N_ROWS).clone();

    assert_eq!(delta0[2], delta0_debug[2]);

    delta0.iter_mut().enumerate().for_each(|(i, delta0_i)| {
        // delta1 = delta0 << 256 1x512
        let delta1_i = [&delta0_i[N_ROWS..], &vec![Point::infinity(); N_ROWS]].concat();
        debug_assert_eq!(delta1_i, delta1_debug[i]);
        q_.push(delta1_i);
    });

    let mut delta1_flat: Vec<_> = q_.into_iter().flatten().collect();

    println!("ecntt batch for delta1");
    //q[mu] = ECNTT_row(delta1) 1x512
    ecntt_batch(&mut delta1_flat, 2 * N_ROWS, 0);

    let q_ = split_vec_to_matrix(&delta1_flat, 2 * N_ROWS).clone();

    assert_eq!(q_, q_debug);

    println!("finall check");

    let P = q_
        .iter()
        .map(|row| list_to_reverse_bit_order(&row.clone()))
        .collect::<Vec<_>>()
        .to_vec();

    //final assertion
    assert_eq!(P, P_debug);

    assert_ne!(P[12][23], Point::zero()); //dummy check
    println!("Branch3 {:0.3?}", o.elapsed());
    println!("success !!!",);
}

#[cfg(test)]
mod tests {
    use super::main_flow;

    #[test]
    fn test_main_flow() {
        main_flow();
    }
}
