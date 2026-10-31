#![deny(clippy::float_arithmetic)]
use mixed_gguf_net::field::field_of;
use mixed_gguf_net::gguf;
use mixed_gguf_net::lattice::{make_gates, totals, CELLS, K};
use mixed_gguf_net::physics::{imprint_taylor_green, solar_wind, SLOTS};
use mixed_gguf_net::sha256::hex_digest;

#[test]
fn sha256_known_vectors() {
    assert_eq!(
        hex_digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex_digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn gates_match_the_repository_construction() {
    let g = make_gates(true);
    assert_eq!(g.len(), 288);
    assert!(g.iter().all(|x| x.lut[40] == 40));
    let mut b = Vec::new();
    for x in &g {
        for c in x.ch {
            b.push(u8::try_from(c).unwrap());
        }
        b.extend_from_slice(&x.lut);
    }
    // byte-identical to the C++ make_collisions with the Taylor-Green pairing rule
    assert_eq!(
        hex_digest(&b),
        "a7fca71ac8a85fd3acbe4188e9c246e98e4beeeaa459f3e77d40c0342aa4bb12"
    );
}

#[test]
fn v1_rule_moves_the_uniform_state() {
    let g = make_gates(false);
    assert_eq!(g.len(), 288);
    assert_eq!(g.iter().filter(|x| x.lut[40] == 40).count(), 0);
}

#[test]
fn reads_the_mythos_blue_star() {
    let raw = std::fs::read("data/ggufs/gguf__acer__ACER-MYTHOS-BLUE.gguf").unwrap();
    assert_eq!(raw.len(), 50_752);
    assert_eq!(
        hex_digest(&raw),
        "3dd92a42160a38c121980957ce9eabbea216678892801d7ea3ed4e99e96c3aae"
    );
    let g = gguf::parse(&raw).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.tensors.len(), 6);
    assert_eq!(g.bytes_read_to, raw.len());
    let f = field_of(&g);
    assert_eq!(f.counts.iter().sum::<usize>(), CELLS);
}

#[test]
fn taylor_green_and_wind_keep_mass_and_energy() {
    let mut f = vec![1u8; SLOTS];
    let t0 = totals(&f);
    let (mx, my) = imprint_taylor_green(&mut f);
    assert!(mx > 0 && my > 0);
    let t1 = totals(&f);
    assert_eq!((t1.mass, t1.e2, t1.pz), (t0.mass, t0.e2, t0.pz));
    let added = solar_wind(&mut f);
    let t2 = totals(&f);
    assert_eq!(
        (t2.mass, t2.e2, t2.py, t2.pz),
        (t1.mass, t1.e2, t1.py, t1.pz)
    );
    assert_eq!(t2.px, t1.px + added);
    assert_eq!(f.len(), CELLS * K);
}
