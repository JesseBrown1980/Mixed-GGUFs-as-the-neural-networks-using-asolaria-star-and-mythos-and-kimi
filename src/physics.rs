//! Taylor-Green vortex (integer square-wave form), the solar wind, and the measurements.

use crate::lattice::{chan, idx, vel, CELLS, K, N};

/// Sign of sin(2*pi*(i+1/2)/N) and cos(2*pi*(i+1/2)/N) from the cell index alone.
pub fn sin_sign(i: usize) -> i64 {
    if i < N / 2 {
        1
    } else {
        -1
    }
}
pub fn cos_sign(i: usize) -> i64 {
    if !(N / 4..3 * N / 4).contains(&i) {
        1
    } else {
        -1
    }
}

/// Moving channels with c_x = +1 paired with their x-mirror (same y, z, same energy).
pub fn x_pairs() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for z in -1..=1 {
        for y in -1..=1 {
            v.push((chan(1, y, z), chan(-1, y, z)));
        }
    }
    v
}
pub fn y_pairs() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for z in -1..=1 {
        for x in -1..=1 {
            v.push((chan(x, 1, z), chan(x, -1, z)));
        }
    }
    v
}

/// Move one trit from `from` to `to` in a cell if the source holds one and the target has room.
fn shift(f: &mut [u8], base: usize, from: usize, to: usize) -> bool {
    if f[base + from] > 0 && f[base + to] < 2 {
        f[base + from] -= 1;
        f[base + to] += 1;
        true
    } else {
        false
    }
}

/// Taylor-Green: u_x ~ sin(x)cos(y)cos(z), u_y ~ -cos(x)sin(y)cos(z), u_z = 0, taken as
/// signs (the square wave). Each mirror pair moves one trit toward the local flow sign.
/// Mass and 2E are unchanged by construction. Returns (x_moves, y_moves).
pub fn imprint_taylor_green(f: &mut [u8]) -> (u64, u64) {
    let xp = x_pairs();
    let yp = y_pairs();
    let (mut mx, mut my) = (0u64, 0u64);
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let base = idx(x, y, z) * K;
                let sx = sin_sign(x) * cos_sign(y) * cos_sign(z);
                let sy = -cos_sign(x) * sin_sign(y) * cos_sign(z);
                for &(p, m) in &xp {
                    let moved = if sx > 0 {
                        shift(f, base, m, p)
                    } else {
                        shift(f, base, p, m)
                    };
                    mx += u64::from(moved);
                }
                for &(p, m) in &yp {
                    let moved = if sy > 0 {
                        shift(f, base, m, p)
                    } else {
                        shift(f, base, p, m)
                    };
                    my += u64::from(moved);
                }
            }
        }
    }
    (mx, my)
}

/// Solar wind: on the plane x = 0, every x-mirror pair moves one trit from -x to +x where
/// it can. Mass and 2E are unchanged; px rises by 2 per move. Returns the px added.
pub fn solar_wind(f: &mut [u8]) -> i64 {
    let xp = x_pairs();
    let mut added = 0i64;
    for z in 0..N {
        for y in 0..N {
            let base = idx(0, y, z) * K;
            for &(p, m) in &xp {
                if shift(f, base, m, p) {
                    added += 2;
                }
            }
        }
    }
    added
}

/// Momentum per cell.
pub fn momentum(f: &[u8]) -> Vec<[i64; 3]> {
    f.chunks_exact(K)
        .map(|cell| {
            let mut j = [0i64; 3];
            for (k, &n) in cell.iter().enumerate() {
                let (x, y, z) = vel(k);
                let n = i64::from(n);
                j[0] += n * x;
                j[1] += n * y;
                j[2] += n * z;
            }
            j
        })
        .collect()
}

/// Enstrophy in integer units: sum over cells of |2 * curl j|^2 with central differences
/// (each derivative is j(i+1) - j(i-1), i.e. twice the gradient).
pub fn enstrophy(j: &[[i64; 3]]) -> i64 {
    let w = |a: usize| (a + N - 1) % N;
    let e = |a: usize| (a + 1) % N;
    let mut s = 0i64;
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let d = |comp: usize, axis: usize| -> i64 {
                    let (p, m) = match axis {
                        0 => (idx(e(x), y, z), idx(w(x), y, z)),
                        1 => (idx(x, e(y), z), idx(x, w(y), z)),
                        _ => (idx(x, y, e(z)), idx(x, y, w(z))),
                    };
                    j[p][comp] - j[m][comp]
                };
                let cx = d(2, 1) - d(1, 2);
                let cy = d(0, 2) - d(2, 0);
                let cz = d(1, 0) - d(0, 1);
                s += cx * cx + cy * cy + cz * cz;
            }
        }
    }
    s
}

/// Slots (cell, channel) equal between two states.
pub fn same_slots(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as u64
}

/// For each periodic distance d = min(x, N - x) from the plane x = 0, whether any slot in
/// that pair of planes differs between two states. Returns the furthest such distance.
pub fn front_distance(a: &[u8], b: &[u8]) -> Option<usize> {
    let mut far: Option<usize> = None;
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let base = idx(x, y, z) * K;
                if a[base..base + K] != b[base..base + K] {
                    let d = x.min(N - x);
                    far = Some(far.map_or(d, |f| f.max(d)));
                }
            }
        }
    }
    far
}

/// px summed per x-plane.
pub fn px_by_plane(j: &[[i64; 3]]) -> [i64; N] {
    let mut out = [0i64; N];
    for (cell, m) in j.iter().enumerate() {
        out[cell % N] += m[0];
    }
    out
}

/// Number of (cell, channel) slots.
pub const SLOTS: usize = CELLS * K;
