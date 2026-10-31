//! D3Q27 ternary lattice gas on an N^3 periodic cube. A port of the gate network in
//! Neural-network-inside-a-neural-network-using-trits-to-resolve-Finiteness
//! (asolaria_ternary_fluid_v1.cpp), with the pairing rule of
//! asolaria_taylor_green_regularity.cpp: code 40 = (1,1,1,1) is a fixed point of every gate.

use std::collections::{BTreeMap, BTreeSet};

pub const N: usize = 64;
pub const K: usize = 27;
pub const CELLS: usize = N * N * N;
pub const REST: usize = 13;
pub const GATES_PER_CELL: usize = 32;

/// Channel k = (z+1)*9 + (y+1)*3 + (x+1), velocities in {-1,0,1}^3.
pub fn vel(k: usize) -> (i64, i64, i64) {
    let x = (k % 3) as i64 - 1;
    let y = ((k / 3) % 3) as i64 - 1;
    let z = (k / 9) as i64 - 1;
    (x, y, z)
}
pub fn chan(x: i64, y: i64, z: i64) -> usize {
    ((z + 1) * 9 + (y + 1) * 3 + (x + 1)) as usize
}
pub fn energy(k: usize) -> i64 {
    let (x, y, z) = vel(k);
    x * x + y * y + z * z
}
pub fn idx(x: usize, y: usize, z: usize) -> usize {
    (z * N + y) * N + x
}
fn wrap(a: i64) -> usize {
    a.rem_euclid(N as i64) as usize
}

#[derive(Clone)]
pub struct Gate {
    pub ch: [usize; 4],
    pub lut: [u8; 81],
}

fn decode(code: usize) -> [i64; 4] {
    [
        (code % 3) as i64,
        ((code / 3) % 3) as i64,
        ((code / 9) % 3) as i64,
        ((code / 27) % 3) as i64,
    ]
}

fn invariant(ch: &[usize; 4], code: usize) -> [i64; 5] {
    let d = decode(code);
    let mut s = [0i64; 5];
    for j in 0..4 {
        let (x, y, z) = vel(ch[j]);
        s[0] += d[j];
        s[1] += d[j] * x;
        s[2] += d[j] * y;
        s[3] += d[j] * z;
        s[4] += d[j] * energy(ch[j]);
    }
    s
}

/// The repository's gate construction: every 4-channel set admitting an exact 2->2
/// collision; within each class of (mass, momentum, 2E) the 81 codes are paired into
/// transpositions, so every gate is an involution. `pin_uniform` keeps code 40 fixed.
pub fn make_gates(pin_uniform: bool) -> Vec<Gate> {
    // the C++ indexes velocities x fastest from -1: same order as `vel` here
    let mut pairs: BTreeMap<(i64, i64, i64, i64), Vec<[usize; 2]>> = BTreeMap::new();
    for a in 0..K {
        for b in a + 1..K {
            let (ax, ay, az) = vel(a);
            let (bx, by, bz) = vel(b);
            pairs
                .entry((ax + bx, ay + by, az + bz, energy(a) + energy(b)))
                .or_default()
                .push([a, b]);
        }
    }
    let mut quads: BTreeSet<[usize; 4]> = BTreeSet::new();
    for p in pairs.values() {
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                let mut q = [p[i][0], p[i][1], p[j][0], p[j][1]];
                q.sort_unstable();
                if q[0] != q[1] && q[1] != q[2] && q[2] != q[3] {
                    quads.insert(q);
                }
            }
        }
    }
    let mut out = Vec::new();
    for q in quads {
        let mut lut = [0u8; 81];
        let mut cls: BTreeMap<[i64; 5], Vec<usize>> = BTreeMap::new();
        for (code, slot) in lut.iter_mut().enumerate() {
            *slot = code as u8;
            cls.entry(invariant(&q, code)).or_default().push(code);
        }
        for codes in cls.values() {
            let kept: Vec<usize> = codes
                .iter()
                .copied()
                .filter(|&c| !(pin_uniform && c == 40))
                .collect();
            for pair in kept.chunks_exact(2) {
                lut[pair[0]] = pair[1] as u8;
                lut[pair[1]] = pair[0] as u8;
            }
        }
        let swaps = lut
            .iter()
            .enumerate()
            .filter(|(i, &v)| *i != usize::from(v))
            .count();
        for code in 0..81 {
            let dst = usize::from(lut[code]);
            assert_eq!(usize::from(lut[dst]), code, "gate not involutive");
            assert_eq!(
                invariant(&q, code),
                invariant(&q, dst),
                "gate breaks conservation"
            );
        }
        if swaps > 0 {
            out.push(Gate { ch: q, lut });
        }
    }
    out
}

fn gate_index(step: usize, cell: usize, j: usize, total: usize) -> usize {
    (step * GATES_PER_CELL + cell * 17 + j * 37) % total
}

fn apply_one(f: &mut [u8], base: usize, g: &Gate) {
    let code = usize::from(f[base + g.ch[0]])
        + 3 * usize::from(f[base + g.ch[1]])
        + 9 * usize::from(f[base + g.ch[2]])
        + 27 * usize::from(f[base + g.ch[3]]);
    let d = usize::from(g.lut[code]);
    f[base + g.ch[0]] = (d % 3) as u8;
    f[base + g.ch[1]] = ((d / 3) % 3) as u8;
    f[base + g.ch[2]] = ((d / 9) % 3) as u8;
    f[base + g.ch[3]] = ((d / 27) % 3) as u8;
}

pub fn collide(f: &mut [u8], gates: &[Gate], step: usize) {
    let total = gates.len();
    for cell in 0..CELLS {
        let base = cell * K;
        for j in 0..GATES_PER_CELL {
            apply_one(f, base, &gates[gate_index(step, cell, j, total)]);
        }
    }
}

pub fn uncollide(f: &mut [u8], gates: &[Gate], step: usize) {
    let total = gates.len();
    for cell in 0..CELLS {
        let base = cell * K;
        for j in (0..GATES_PER_CELL).rev() {
            apply_one(f, base, &gates[gate_index(step, cell, j, total)]);
        }
    }
}

/// n_k(x + c_k, t+1) = n_k(x, t)
pub fn stream(cur: &[u8], next: &mut [u8]) {
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let src = idx(x, y, z) * K;
                for k in 0..K {
                    let (cx, cy, cz) = vel(k);
                    let dst = idx(
                        wrap(x as i64 + cx),
                        wrap(y as i64 + cy),
                        wrap(z as i64 + cz),
                    ) * K
                        + k;
                    next[dst] = cur[src + k];
                }
            }
        }
    }
}

pub fn unstream(after: &[u8], before: &mut [u8]) {
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let dst = idx(x, y, z) * K;
                for k in 0..K {
                    let (cx, cy, cz) = vel(k);
                    let src = idx(
                        wrap(x as i64 + cx),
                        wrap(y as i64 + cy),
                        wrap(z as i64 + cz),
                    ) * K
                        + k;
                    before[dst + k] = after[src];
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Totals {
    pub mass: i64,
    pub px: i64,
    pub py: i64,
    pub pz: i64,
    pub e2: i64,
}

pub fn totals(f: &[u8]) -> Totals {
    let mut t = Totals::default();
    for cell in f.chunks_exact(K) {
        for (k, &n) in cell.iter().enumerate() {
            let n = i64::from(n);
            let (x, y, z) = vel(k);
            t.mass += n;
            t.px += n * x;
            t.py += n * y;
            t.pz += n * z;
            t.e2 += n * energy(k);
        }
    }
    t
}

/// Per-channel masses: the 27 outer shell modes.
pub fn modes(f: &[u8]) -> [i64; K] {
    let mut m = [0i64; K];
    for cell in f.chunks_exact(K) {
        for (k, &n) in cell.iter().enumerate() {
            m[k] += i64::from(n);
        }
    }
    m
}
