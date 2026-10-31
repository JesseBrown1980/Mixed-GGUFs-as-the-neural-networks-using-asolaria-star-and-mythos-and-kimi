//! One GGUF -> one 64^3 trit field (x fastest). Integer only.
//!
//! 1. the file has 64^3 tensors: per-cell maximum magnitude over them
//! 2. else it has 64x64 slices: z-layer z takes slice (z mod S)
//! 3. else: every element of every tensor, in file order, tiled
//!
//! Trit: 0 if magnitude 0; 1 if <= the lower median of the file's non-zero magnitudes; else 2.

use crate::gguf::Gguf;
use crate::lattice::{CELLS, N};

pub struct Field {
    pub trits: Vec<u8>,
    pub layout: String,
    pub median: u32,
    pub counts: [usize; 3],
}

pub fn field_of(g: &Gguf) -> Field {
    let cube = (N * N * N) as u64;
    let n = N as u64;
    let cubes: Vec<&Vec<u32>> = g
        .tensors
        .iter()
        .filter(|t| {
            t.dims.len() == 3 && t.dims.iter().all(|&d| d == n) && t.mags.len() as u64 == cube
        })
        .map(|t| &t.mags)
        .collect();
    let slices: Vec<&Vec<u32>> = g
        .tensors
        .iter()
        .filter(|t| t.dims.len() == 2 && t.dims[0] == n && t.dims[1] == n)
        .map(|t| &t.mags)
        .collect();
    let (mags, layout): (Vec<u32>, String) = if !cubes.is_empty() {
        let mut m = vec![0u32; CELLS];
        for c in &cubes {
            for (a, &b) in m.iter_mut().zip(c.iter()) {
                *a = (*a).max(b);
            }
        }
        (m, format!("max_of_{}_cubes", cubes.len()))
    } else if !slices.is_empty() {
        let mut m = Vec::with_capacity(CELLS);
        for z in 0..N {
            m.extend_from_slice(slices[z % slices.len()]);
        }
        (m, format!("{}_slices_on_z", slices.len()))
    } else {
        let stream: Vec<u32> = g
            .tensors
            .iter()
            .flat_map(|t| t.mags.iter().copied())
            .collect();
        let m: Vec<u32> = (0..CELLS).map(|i| stream[i % stream.len()]).collect();
        (m, format!("stream_of_{}_tiled", stream.len()))
    };
    let mut nz: Vec<u32> = mags.iter().copied().filter(|&v| v > 0).collect();
    nz.sort_unstable();
    let median = if nz.is_empty() {
        0
    } else {
        nz[(nz.len() - 1) / 2]
    };
    let mut counts = [0usize; 3];
    let trits: Vec<u8> = mags
        .iter()
        .map(|&v| {
            let t: u8 = if v == 0 {
                0
            } else if v <= median {
                1
            } else {
                2
            };
            counts[usize::from(t)] += 1;
            t
        })
        .collect();
    Field {
        trits,
        layout,
        median,
        counts,
    }
}
