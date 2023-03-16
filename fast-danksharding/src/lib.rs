pub mod fast_danksharding;
pub mod matrix;
pub mod utils;

use std::ffi::c_int;

use icicle_utils::field::*;

use crate::{
    fast_danksharding::{M_POINTS, N_ROWS},
    matrix::split_vec_to_matrix,
    utils::{csv_be_to_u32_be_limbs, from_limbs},
};

extern "C" {
    fn sum_of_points(
        out: *mut Point,
        input: *const Point,
        nof_rows: usize,
        nof_cols: usize,
        l: usize,
        device_id: usize,
    ) -> c_int;
}

pub fn addp_vec(
    a: &mut [Point],
    b: &[Point],
    nof_rows: usize,
    nof_cols: usize,
    l: usize,
    device_id: usize,
) -> i32 {
    unsafe {
        sum_of_points(
            a as *mut _ as *mut Point,
            b as *const _ as *const Point,
            nof_rows,
            nof_cols,
            l,
            device_id,
        )
    }
}

pub fn get_debug_data_scalars(filename: &str, height: usize, lenght: usize) -> Vec<Vec<Scalar>> {
    let data_root_path = get_test_set_path();

    let limbs = csv_be_to_u32_be_limbs(&format!("{}{}", data_root_path, filename), SCALAR_LIMBS);

    let result = split_vec_to_matrix(
        &from_limbs(limbs, SCALAR_LIMBS, Scalar::from_limbs_be),
        lenght,
    );
    assert_eq!(result.len(), height);
    assert_eq!(result[0].len(), lenght);

    result
}

fn get_test_set_path() -> String {
    #[cfg(test)]
    let data_root_path = format!("../test_vectors/{}x{}/", N_ROWS, M_POINTS);
    #[cfg(not(test))]
    let data_root_path = format!(
        "{}/test_vectors/{}x{}/",
        std::env::current_dir().unwrap().to_str().unwrap(),
        N_ROWS,
        M_POINTS
    );
    data_root_path
}

fn get_debug_data_points_xy1(filename: &str, height: usize, lenght: usize) -> Vec<Vec<Point>> {
    get_debug_data_points(filename, height, lenght, Point::from_xy1_be_limbs, 2)
}

fn get_debug_data_points<T: Copy + Clone>(
    filename: &str,
    height: usize,
    lenght: usize,
    f: fn(&[u32]) -> T,
    n_base_fields: usize,
) -> Vec<Vec<T>> {
    let data_root_path = get_test_set_path(); //TODO: relative path

    let limbs = csv_be_to_u32_be_limbs(&format!("{}{}", data_root_path, filename), BASE_LIMBS);

    println!("total limbs u32 {:?}", limbs.len());

    let result = split_vec_to_matrix(&from_limbs(limbs, n_base_fields * BASE_LIMBS, f), lenght);
    assert_eq!(result.len(), height);
    assert_eq!(result[0].len(), lenght);

    result
}

pub fn get_debug_data_points_proj_xy1_vec(filename: &str, lenght: usize) -> Vec<Point> {
    get_debug_data_points_xy1(filename, 1, lenght)[0].to_vec()
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Fr, FrParameters};
    use ark_ec::msm::VariableBaseMSM;
    use ark_ff::{BigInteger256, FpParameters};
    use icicle_utils::{msm, mult_sc_vec, multp_vec};

    use crate::{
        fast_danksharding::{M_POINTS, N_ROWS},
        utils::*,
        *,
    };

    #[test]
    #[allow(non_snake_case)]
    fn test_csv_msm() {
        //test debug csv data
        ////0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
        let b = hex_be_to_padded_u32_be_vec(
            "17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
            BASE_LIMBS,
        );

        let p = Point::from_xy1_be_limbs(&b);
        assert!(p.to_ark_affine().is_on_curve());

        let x_bf_limbs = hex_be_to_padded_u32_le_vec("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb", BASE_LIMBS);
        let x_bf = BaseField::from_limbs(&x_bf_limbs);

        assert_eq!(p.x, x_bf, "{:08X?} != {:08X?}", p.x, x_bf_limbs);

        let C_rows_debug = get_debug_data_scalars("C_rows.csv", N_ROWS, M_POINTS);
        let SRS = get_debug_data_points_proj_xy1_vec("SRS.csv", M_POINTS);
        let K0 = get_debug_data_points_proj_xy1_vec("K0.csv", N_ROWS);

        let scalars = C_rows_debug[0].to_vec();
        let points_r = SRS.to_vec();

        assert_eq!(p, SRS[0]);

        assert_eq!(points_r.len(), scalars.len());
        for i in 0..points_r.len() {
            assert!(points_r[i].to_ark_affine().is_on_curve())
        }

        assert!(points_r[0].to_ark_affine().is_on_curve());

        let points: Vec<PointAffineNoInfinity> =
            points_r.into_iter().map(|p| p.to_xy_strip_z()).collect();

        let msm_result = msm(&points, &scalars, 0);

        assert_eq!(msm_result, K0[0]);

        let point_r_ark: Vec<_> = points.iter().map(|x| x.to_ark_repr()).collect();
        let scalars_r_ark: Vec<_> = scalars.iter().map(|x| x.to_ark_mod_p().0).collect();

        let msm_result_ark = VariableBaseMSM::multi_scalar_mul(&point_r_ark, &scalars_r_ark);

        assert_eq!(msm_result.to_ark_affine(), msm_result_ark);

        assert_eq!(msm_result.to_ark(), msm_result_ark);
        assert_eq!(msm_result, Point::from_ark(msm_result_ark));
    }

    #[test]
    fn test_parse_csv_point() {
        // bls12-381 generator
        let b = hex_be_to_padded_u32_be_vec(
                "17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1",
                BASE_LIMBS,
            );

        let p = Point::from_xy1_be_limbs(&b);

        assert!(p.to_ark_affine().is_on_curve());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_arc_py_scalar() {
        let py_str = "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";

        let limbs = hex_be_to_padded_u32_le_vec(py_str, SCALAR_LIMBS);

        let sc = Scalar::from_limbs_le(&limbs);

        let modulus = BigInteger256::new([
            0xffffffff00000001,
            0x53bda402fffe5bfe,
            0x3339d80809a1d805,
            0x73eda753299d7d48,
        ]);

        assert_eq!(modulus, FrParameters::MODULUS);

        let ark_mod = Fr::new(modulus);
        assert_eq!(
            sc.to_ark_mod_p(),
            ark_mod,
            "\n******\n{:08X?}\n******\n{:08X?}\n******\n",
            sc.to_ark_mod_p(),
            ark_mod
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_csv_scalar() {
        let sample_be = "12039d72be179e9cf6d56ce4a99bd4e8b8ff59367c57bf66f094d4143f7d4ade";
        assert_eq!(64, sample_be.len());
        let limbs = hex_be_to_padded_u32_be_vec(sample_be, SCALAR_LIMBS);

        assert_eq!(limbs.len(), SCALAR_LIMBS);
        let scalar = Scalar::from_limbs_be(&limbs);

        // let sample_u64le = [
        //     0xf094d4143f7d4adeu64,
        //     0xb8ff59367c57bf66u64,
        //     0xf6d56ce4a99bd4e8u64,
        //     0x12039d72be179e9cu64,
        // ];

        let sample_u32le = [
            0x3f7d4adeu32,
            0xf094d414u32,
            0x7c57bf66u32,
            0xb8ff5936u32,
            0xa99bd4e8u32,
            0xf6d56ce4u32,
            0xbe179e9cu32,
            0x12039d72u32,
        ];

        assert_eq!(scalar.limbs(), sample_u32le)
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_saclar_mul() {
        let mut intoo = [Scalar::one(), Scalar::one(), Scalar::zero()];
        let expected = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        mult_sc_vec(&mut intoo, &expected, 0);
        assert_eq!(intoo, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_equality() {
        let left = Point::zero();
        let right = Point::zero();
        assert_eq!(left, right);
        let right = Point::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_point_mul() {
        let dummy_one = Point {
            x: BaseField::one(),
            y: BaseField::zero(),
            z: BaseField::one(),
        };

        let mut inout = [dummy_one, dummy_one, Point::zero()];
        let scalars = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        let expected = [
            Point::zero(),
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
        ];
        multp_vec(&mut inout, &scalars, 0);
        assert_eq!(inout, expected);
    }
}
