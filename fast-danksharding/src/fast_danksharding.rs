use std::time::Instant;

use icicle::curves::bls12_381::*;

use icicle::test_bls12_381::mult_sc_vec_bls12_381 as mult_sc_vec;
use icicle::test_bls12_381::mult_sc_batch_vec_bls12_381 as mult_sc_batch_vec;
use icicle::test_bls12_381::ntt_batch_bls12_381 as ntt_batch;
use icicle::test_bls12_381::intt_batch_bls12_381 as intt_batch;
use icicle::test_bls12_381::ecntt_batch_bls12_381 as ecntt_batch;
use icicle::test_bls12_381::iecntt_batch_bls12_381 as iecntt_batch;
use icicle::test_bls12_381::msm_batch_bls12_381 as msm_batch;
use icicle::test_bls12_381::multp_vec_bls12_381 as multp_vec;

use crate::{matrix::*, utils::*, *};

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
    let D_in = get_debug_data_scalar_vec("D_in.csv");
    let tf_u = &get_debug_data_scalars("roots_u.csv", 1, N_ROWS)[0];
    let SRS = get_debug_data_points_proj_xy1_vec("SRS.csv", M_POINTS);
    let roots_w = get_debug_data_scalars("roots_w.csv", M_POINTS, 1);
    let tf_w = rows_to_cols(&roots_w)[0].to_vec().to_vec();
    //TODO: now S is preprocessed, copy preprocessing here
    let S = get_debug_data_points_proj_xy1_vec("S.csv", 2 * M_POINTS);

    let mut q_ = Vec::<Vec<Point_BLS12_381>>::new();
    const l: usize = 16;
    println!("loaded test data, processing...");

    //TODO: branches and many steps can be parallel
    ///////////////////
    let mut C_rows = D_in.clone();
    let pre_time = Instant::now();

    //C_rows = INTT_rows(D_in)
    intt_batch(&mut C_rows, M_POINTS, 0);
    assert_eq!(C_rows[17], get_debug_data_scalar_vec("C_rows.csv")[17]);

    //[s] = SRS 1x4096
    assert!(SRS[0].to_ark_affine().is_on_curve());
    let s_affine: Vec<_> = SRS.iter().map(|p| p.to_xy_strip_z()).collect();
    println!("pre-branch {:0.3?}", pre_time.elapsed());

    ////////////////////////////////
    println!("Branch 1");
    ////////////////////////////////
    let br1_time = Instant::now();

    // K0 = MSM_rows(C_rows) (256x1)
    let K0 = msm_batch(&vec![s_affine; N_ROWS].concat(), &C_rows, N_ROWS, 0);
    println!("K0 {:0.3?}", br1_time.elapsed());
    debug_assert_eq!(K0, get_debug_data_points_proj_xy1_vec("K0.csv", N_ROWS));

    // B0 = ECINTT_col(K0) N_ROWS x 1 (256x1)
    let mut B0 = K0.clone();
    iecntt_batch(&mut B0, N_ROWS, 0);
    println!("B0 {:0.3?}", br1_time.elapsed());
    debug_assert_eq!(B0, get_debug_data_points_proj_xy1_vec("B0.csv", N_ROWS));

    // B1 = MUL_col(B0, [1 u u^2 ...]) N_ROWS x 1 (256x1)
    let mut B1 = B0.clone();
    multp_vec(&mut B1, &tf_u, 0);
    println!("B1 {:0.3?}", br1_time.elapsed());
    debug_assert_eq!(B1, get_debug_data_points_proj_xy1_vec("B1.csv", N_ROWS));

    // K1 = ECNTT_col(B1) N_ROWS x 1 (256x1)
    let mut K1 = B1;
    ecntt_batch(&mut K1,  N_ROWS, 0);
    println!("K1 {:0.3?}", br1_time.elapsed());
    debug_assert_eq!(K1, get_debug_data_points_proj_xy1_vec("K1.csv", N_ROWS));

    // K = [K0, K1]  // 2*N_ROWS x 1 (512x1 commitments)
    let K = [K0, K1].concat();
    println!("K {:0.3?}", br1_time.elapsed());
    
    println!("Branch1 {:0.3?}", br1_time.elapsed());
    debug_assert_eq!(K, get_debug_data_points_proj_xy1_vec("K.csv", 2 * N_ROWS));

    ////////////////////////////////
    println!("Branch 2");
    ////////////////////////////////
    let br2_time = Instant::now();

    // C = INTT_cols(C_rows) 256x4096 col
    let mut C: Vec<_> = rows_to_cols_flatten(&C_rows, M_POINTS);
    intt_batch(&mut C, N_ROWS, 0);
    let C = rows_to_cols_flatten(&C, N_ROWS);
    println!("C {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(C, get_debug_data_scalar_vec("C.csv"));

    // C0 = MUL_cols(C, [1 w w^2 ...]) 256x4096
    let mut C0 = C.clone();
    mult_sc_batch_vec(&mut C0, &tf_w, 0);
    println!("C0 {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(C0, get_debug_data_scalar_vec("C0.csv"));

    // C2 = MUL_rows(C, [1 u u^2 ...]) 256x4096
    let mut C2 = rows_to_cols_flatten(&C, M_POINTS);
    mult_sc_batch_vec(&mut C2, &tf_u, 0);
    let C2 = rows_to_cols_flatten(&C2, N_ROWS);
    println!("C2 {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(C2, get_debug_data_scalar_vec("C2.csv"));

    //E0 = NTT_rows(C0) 256x4096
    let mut E0: Vec<_> = C0;
    ntt_batch(&mut E0, M_POINTS, 0);
    println!("E0 {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(E0, get_debug_data_scalar_vec("E0.csv"));

    //E1 = MUL_rows(E0, [1 u u^2 ...]) 256x4096
    let mut E1 = rows_to_cols_flatten(&E0, M_POINTS);
    mult_sc_batch_vec(&mut E1, &tf_u, 0);
    let E1 = rows_to_cols_flatten(&E1, N_ROWS);
    println!("E1 {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(E1, get_debug_data_scalar_vec("E1.csv"));

    // E2 = NTT_rows(C2) 256x4096
    let mut E2: Vec<_> = C2;
    ntt_batch(&mut E2, M_POINTS, 0);
    println!("E2 {:0.3?}", br2_time.elapsed());
    debug_assert_eq!(E2, get_debug_data_scalar_vec("E2.csv"));

    //D_rows = NTT_cols(E0) 256x4096
    let mut D_rows = rows_to_cols_flatten(&E0, M_POINTS);
    ntt_batch(&mut D_rows, N_ROWS, 0);
    let D_rows = rows_to_cols_flatten(&D_rows, N_ROWS);

    debug_assert_eq!(D_rows, get_debug_data_scalar_vec("D_rows.csv"));

    // D_both = NTT_cols(E1) 256x4096
    let mut D_both = rows_to_cols_flatten(&E1, M_POINTS);
    ntt_batch(&mut D_both, N_ROWS, 0);
    let D_both = rows_to_cols_flatten(&D_both, N_ROWS);

    debug_assert_eq!(D_both, get_debug_data_scalar_vec("D_both.csv"));

    // D_cols = NTT_cols(E2) 256x4096
    let mut D_cols = rows_to_cols_flatten(&E2, M_POINTS);
    ntt_batch(&mut D_cols, N_ROWS, 0);
    let D_cols = rows_to_cols_flatten(&D_cols, N_ROWS);

    debug_assert_eq!(D_cols, get_debug_data_scalar_vec("D_cols.csv"));

    //D_b4rbo = INTERLEAVE_cols( [D_in; D_cols], [D_rows; D_both] ) 512x8192
    let D_b4rbo = interleave_cols(
        &[
            split_vec_to_matrix(&D_in, M_POINTS),
            split_vec_to_matrix(&D_cols, M_POINTS),
        ]
        .concat(),
        &[
            split_vec_to_matrix(&D_rows, M_POINTS),
            split_vec_to_matrix(&D_both, M_POINTS),
        ]
        .concat(),
    );

    debug_assert_eq!(
        D_b4rbo,
        get_debug_data_scalars("D_b4rbo.csv", 2 * N_ROWS, 2 * M_POINTS)
    );

    debug_assert_eq!(D_b4rbo.len(), N_ROWS * 2);
    debug_assert_eq!(D_b4rbo[0].len(), M_POINTS * 2);

    //D = RBO_cols(D_b4rbo) 512x8192
    let D = D_b4rbo
        .iter()
        .map(|row| list_to_reverse_bit_order(row))
        .collect::<Vec<_>>();

    debug_assert_eq!(D, get_debug_data_scalars("D.csv", 2 * N_ROWS, 2 * M_POINTS));

    println!("Branch2 {:0.3?}", br2_time.elapsed());

    ////////////////////////////////
    println!("Branch 3");

    let D_b4rbo_flat = D_b4rbo.concat();
    let mut d0 = vec![S.clone();  2 * N_ROWS].concat();
    let mut d1 = vec![Point::infinity(); (2 * N_ROWS) * (2 * M_POINTS / l)];

    ////////////////////////////////
    let br3_time = Instant::now();
    multp_vec(&mut d0, &D_b4rbo_flat, 0);
    println!("mult batch d0 {:0.3?}", br3_time.elapsed());
    
    debug_assert_eq!(
        d0,
        get_debug_data_points_proj_xy1_vec("d0.csv", 2 * N_ROWS * 2 * M_POINTS)
    );


    addp_vec(&mut d1, &d0, 2 * N_ROWS, 2 * M_POINTS, l, 0);
    println!("add batch d0 + d1 {:0.3?}", br3_time.elapsed());

    debug_assert_eq!(
        d1,
        get_debug_data_points_proj_xy1_vec("d1.csv", 2 * N_ROWS * 2 * N_ROWS)
    );

    let mut delta0: Vec<_> = d1;
    //delta0 = ECINTT_row(d1) 1x512
    iecntt_batch(&mut delta0, 2 * N_ROWS, 0);
    println!("iecntt batch for delta0 {:0.3?}", br3_time.elapsed());
    debug_assert_eq!(
        delta0,
        get_debug_data_points_proj_xy1_vec("delta0.csv", 2 * N_ROWS * 2 * N_ROWS)
    );

    delta0.chunks_mut(2 * N_ROWS).for_each(|delta0_i| {
        // delta1 = delta0 << 256 1x512
        let delta1_i = [&delta0_i[N_ROWS..], &vec![Point::infinity(); N_ROWS]].concat();
        q_.push(delta1_i);
    });

    let mut delta1: Vec<_> = q_.into_iter().flatten().collect();

    //q[mu] = ECNTT_row(delta1) 1x512
    ecntt_batch(&mut delta1, 2 * N_ROWS, 0);
    println!("ecntt batch for delta1 {:0.3?}", br3_time.elapsed());

    let q_ = split_vec_to_matrix(&delta1, 2 * N_ROWS);

    debug_assert_eq!(
        q_,
        get_debug_data_points_xy1("q.csv", 2 * N_ROWS, 2 * N_ROWS)
    );

    println!("final check");

    let P = q_
        .iter()
        .map(|row| list_to_reverse_bit_order(&row.clone()))
        .collect::<Vec<_>>()
        .to_vec();

    //final assertion
    println!("Branch3 {:0.3?}", br3_time.elapsed());

    assert_eq!(
        P,
        get_debug_data_points_xy1("P.csv", 2 * N_ROWS, 2 * N_ROWS)
    );

    assert_ne!(P[12][23], Point_BLS12_381::zero()); //dummy check
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
