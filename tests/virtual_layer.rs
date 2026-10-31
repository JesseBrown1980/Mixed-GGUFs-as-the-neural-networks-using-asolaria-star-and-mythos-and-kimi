#![deny(clippy::float_arithmetic)]
use mixed_gguf_net::virtual_layer::{
    corr_permille, diag_part, gram, isqrt, linear, overhead, quad, sign_classes, weights,
};

#[test]
fn twenty_six_weights_thirteen_sign_classes() {
    let w = weights();
    assert_eq!(w.len(), 26);
    assert!(w.iter().all(|c| c.iter().all(|&x| (-1..=1).contains(&x))));
    assert_eq!(sign_classes().len(), 13);
    assert_eq!(overhead(&[1, -1, 1]), 3);
    assert_eq!(linear(&[1, -1, 0], &[10, 4, 99]), 6);
}

#[test]
fn quadratic_readout_equals_the_readout_of_the_combined_field() {
    let a = vec![[3i64, -1, 2], [0, 5, -4], [7, 7, 7]];
    let b = vec![[1i64, 1, 1], [-2, 0, 3], [4, -5, 6]];
    let c = vec![[0i64, 2, -1], [9, -9, 1], [-3, 8, 2]];
    let g = gram([&a, &b, &c]);
    for w in weights() {
        let combined: i128 = (0..3)
            .map(|cell| {
                let v: Vec<i64> = (0..3)
                    .map(|comp| w[0] * a[cell][comp] + w[1] * b[cell][comp] + w[2] * c[cell][comp])
                    .collect();
                i128::from(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            })
            .sum();
        assert_eq!(quad(&w, &g), combined);
        assert!(quad(&w, &g) >= 0);
    }
    assert_eq!(diag_part(&[1, 1, 1], &g), g[0][0] + g[1][1] + g[2][2]);
}

#[test]
fn integer_sqrt_and_correlation() {
    for n in [0u128, 1, 2, 3, 4, 15, 16, 17, 1_000_000, 999_999_999_999] {
        let r = isqrt(n);
        assert!(r * r <= n && (r + 1) * (r + 1) > n);
    }
    let a = vec![[1i64, 2, 3]];
    let g = gram([&a, &a, &[[-1i64, -2, -3]]]);
    assert_eq!(corr_permille(&g, 0, 1), 1000);
    assert_eq!(corr_permille(&g, 0, 2), -1000);
}
