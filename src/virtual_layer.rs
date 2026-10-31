//! The virtual layer: weights c in {-1, 0, +1}^3 over the three lattices
//! (asolaria, shuffled, uniform). A virtual state sum_i c_i * state_i is not a lattice
//! state, but every readout of it is exact:
//! - linear readouts are sum_i c_i * readout_i;
//! - quadratic readouts of a field F are c^T G c, with G_ik = sum_cells F_i . F_k.
//!
//! No sampling: all 26 non-zero weight vectors are evaluated exactly.

pub const B: usize = 3;
pub type Gram = [[i128; B]; B];

/// The 26 non-zero balanced-trit weight vectors.
pub fn weights() -> Vec<[i64; B]> {
    let mut v = Vec::new();
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                if (a, b, c) != (0, 0, 0) {
                    v.push([a, b, c]);
                }
            }
        }
    }
    v
}

/// One representative per sign class (first non-zero weight is +1): 13 vectors.
pub fn sign_classes() -> Vec<[i64; B]> {
    weights()
        .into_iter()
        .filter(|c| c.iter().find(|&&x| x != 0) == Some(&1))
        .collect()
}

/// Overhead ||c||_1, the cost the virtual QNN pays in samples; here it is only a label.
pub fn overhead(c: &[i64; B]) -> i64 {
    c.iter().map(|x| x.abs()).sum()
}

pub fn linear(c: &[i64; B], v: &[i64; B]) -> i64 {
    c.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}

/// Gram matrix of three vector fields.
pub fn gram(f: [&[[i64; 3]]; B]) -> Gram {
    let mut g = [[0i128; B]; B];
    for i in 0..B {
        for k in i..B {
            let s: i128 = f[i]
                .iter()
                .zip(f[k].iter())
                .map(|(a, b)| i128::from(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
                .sum();
            g[i][k] = s;
            g[k][i] = s;
        }
    }
    g
}

pub fn quad(c: &[i64; B], g: &Gram) -> i128 {
    let mut s = 0i128;
    for i in 0..B {
        for k in 0..B {
            s += i128::from(c[i]) * i128::from(c[k]) * g[i][k];
        }
    }
    s
}

/// sum_i c_i^2 G_ii: what c^T G c would be if the fields were uncorrelated.
pub fn diag_part(c: &[i64; B], g: &Gram) -> i128 {
    (0..B).map(|i| i128::from(c[i] * c[i]) * g[i][i]).sum()
}

pub fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Correlation of fields i and k in per-mille, rounded toward zero.
pub fn corr_permille(g: &Gram, i: usize, k: usize) -> i128 {
    let den = isqrt((g[i][i].max(0) as u128) * (g[k][k].max(0) as u128));
    if den == 0 {
        0
    } else {
        g[i][k] * 1000 / den as i128
    }
}
