//! mixed-gguf-net: the 27 Asolaria GGUFs as the trits of the nested ternary network,
//! put into a Taylor-Green vortex and pushed by a solar wind. Integer only.
//!
//! usage: mixed-gguf-net all DATA_DIR OUT_DIR [TICKS]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

use mixed_gguf_net::field::field_of;
use mixed_gguf_net::gguf;
use mixed_gguf_net::hbp::{index, Hbp};
use mixed_gguf_net::lattice::{
    collide, make_gates, modes, stream, totals, uncollide, unstream, vel, Gate, Totals, CELLS, K,
    N, REST,
};
use mixed_gguf_net::physics::{
    enstrophy, front_distance, imprint_taylor_green, momentum, px_by_plane, same_slots, solar_wind,
    SLOTS,
};
use mixed_gguf_net::sha256::{digest, hex, hex_digest};
use std::fs;
use std::path::Path;

const P_MOD: i64 = 1_000_081;
const TEN_TRILLION: i64 = 10_000_000_000_000;

fn field_value(line: &str, key: &str) -> Option<String> {
    line.split('|')
        .find_map(|kv| kv.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
        .map(str::to_string)
}

fn write(out: &Path, name: &str, body: &str, files: &mut Vec<(String, usize, String)>) {
    let p = out.join(name);
    fs::write(&p, body).expect("write");
    files.push((name.to_string(), body.len(), hex_digest(body.as_bytes())));
}

/// Build the lattice from the manifest. Returns (lattice, CHANNELS receipt).
fn build(data: &Path) -> (Vec<u8>, Hbp) {
    let manifest = fs::read_to_string(data.join("MANIFEST.hbp")).expect("manifest");
    let mut rows: Vec<(usize, String, String, usize)> = Vec::new();
    for line in manifest.lines().filter(|l| l.starts_with("GGUF|")) {
        let order: usize = field_value(line, "order")
            .and_then(|v| v.parse().ok())
            .expect("order");
        let file = field_value(line, "file").expect("file");
        let sha = field_value(line, "sha256").expect("sha");
        let bytes: usize = field_value(line, "bytes")
            .and_then(|v| v.parse().ok())
            .expect("bytes");
        rows.push((order, file, sha, bytes));
    }
    rows.sort();
    assert_eq!(rows.len(), K, "need 27 GGUFs, one per channel");
    let herself = rows
        .iter()
        .position(|r| r.1.ends_with("ASOLARIA-HERSELF.gguf"))
        .expect("ASOLARIA-HERSELF.gguf");
    let mut channel_of = vec![0usize; K];
    channel_of[herself] = REST;
    let mut free = (0..K).filter(|&k| k != REST);
    for (i, slot) in channel_of.iter_mut().enumerate() {
        if i != herself {
            *slot = free.next().expect("channel");
        }
    }
    let mut lattice = vec![0u8; SLOTS];
    let mut rec = Hbp::new();
    let mut table: Vec<(usize, String)> = Vec::new();
    for (i, (order, file, sha, bytes)) in rows.iter().enumerate() {
        let raw = fs::read(data.join("ggufs").join(file)).expect("gguf");
        assert_eq!(raw.len(), *bytes, "{file}: byte count");
        assert_eq!(&hex_digest(&raw), sha, "{file}: sha256");
        let g = gguf::parse(&raw).unwrap_or_else(|e| panic!("{file}: {e}"));
        let f = field_of(&g);
        let k = channel_of[i];
        for (cell, &t) in f.trits.iter().enumerate() {
            lattice[cell * K + k] = t;
        }
        let (cx, cy, cz) = vel(k);
        table.push((
            k,
            format!(
                "CHANNEL|k={k}|c={cx},{cy},{cz}|order={order}|gguf={file}|sha256={sha}|gguf_version={}|tensors={}|layout={}|median_bits={}|t0={}|t1={}|t2={}|json=0",
                g.version,
                g.tensors.len(),
                f.layout,
                f.median,
                f.counts[0],
                f.counts[1],
                f.counts[2]
            ),
        ));
    }
    table.sort();
    let t = totals(&lattice);
    rec.row(
        "LATTICE",
        &[
            ("dims", format!("{N}x{N}x{N}")),
            ("cells", CELLS.to_string()),
            ("trits", SLOTS.to_string()),
            ("mass", t.mass.to_string()),
            ("px", t.px.to_string()),
            ("py", t.py.to_string()),
            ("pz", t.pz.to_string()),
            ("e2", t.e2.to_string()),
            ("sha256", hex_digest(&lattice)),
        ],
    );
    let mut text = rec.text();
    for (_, line) in table {
        text.push_str(&line);
        text.push('\n');
    }
    let mut r2 = Hbp::new();
    for line in text.lines() {
        r2.row(line.trim_end_matches("|json=0"), &[]);
    }
    (lattice, r2)
}

/// Each channel's cells permuted by a deterministic integer Fisher-Yates (xorshift64,
/// seeded from SHA-256("SHUFFLE|k")). Per-channel trit counts are unchanged.
fn shuffled(lat: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; SLOTS];
    for k in 0..K {
        let d = digest(format!("SHUFFLE|{k}").as_bytes());
        let mut s = u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]) | 1;
        let mut perm: Vec<usize> = (0..CELLS).collect();
        for i in (1..CELLS).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let j = (s % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        for (cell, &src) in perm.iter().enumerate() {
            out[cell * K + k] = lat[src * K + k];
        }
    }
    out
}

fn tick(f: &mut Vec<u8>, scratch: &mut Vec<u8>, gates: &[Gate], t: usize) {
    collide(f, gates, t);
    stream(f, scratch);
    std::mem::swap(f, scratch);
}

/// Hash of the plane x = `x0` (all y, z, all 27 channels): a local observer's view.
fn plane_hash(f: &[u8], x0: usize) -> [u8; 32] {
    let mut buf = Vec::with_capacity(N * N * K);
    for z in 0..N {
        for y in 0..N {
            let b = mixed_gguf_net::lattice::idx(x0, y, z) * K;
            buf.extend_from_slice(&f[b..b + K]);
        }
    }
    digest(&buf)
}

struct RunSummary {
    name: String,
    injected_first64: i64,
    injected_last64: i64,
    injected_total: i64,
    carried_past8_permille: i64,
    ens_off_end: i64,
    ens_on_end: i64,
    ens_start: i64,
    inj_series: Vec<i64>,
    snaps: Vec<Snap>,
}

/// A picture of the run at one tick: wind-on density and (on - off) px, both summed over z.
struct Snap {
    t: usize,
    density: Vec<i64>,
    dpx: Vec<i64>,
}

fn snap(t: usize, off: &[u8], on: &[u8]) -> Snap {
    let mut density = vec![0i64; N * N];
    let mut dpx = vec![0i64; N * N];
    for (cell, (a, b)) in off.chunks_exact(K).zip(on.chunks_exact(K)).enumerate() {
        let xy = cell % (N * N);
        for k in 0..K {
            let (cx, _, _) = vel(k);
            density[xy] += i64::from(b[k]);
            dpx[xy] += (i64::from(b[k]) - i64::from(a[k])) * cx;
        }
    }
    Snap { t, density, dpx }
}

#[allow(clippy::too_many_lines)]
fn run_lattice(
    name: &str,
    gguf_lattice: &[u8],
    gates: &[Gate],
    ticks: usize,
    rec: &mut Hbp,
    series: &mut Hbp,
) -> RunSummary {
    let mut base = gguf_lattice.to_vec();
    let before = totals(&base);
    let (mx, my) = imprint_taylor_green(&mut base);
    let t0 = totals(&base);
    assert_eq!(
        (t0.mass, t0.e2),
        (before.mass, before.e2),
        "imprint must keep mass and 2E"
    );
    let ens0 = enstrophy(&momentum(&base));
    rec.row(
        "TAYLOR_GREEN",
        &[
            ("lattice", name.into()),
            ("x_moves", mx.to_string()),
            ("y_moves", my.to_string()),
            ("mass", t0.mass.to_string()),
            ("px", t0.px.to_string()),
            ("py", t0.py.to_string()),
            ("pz", t0.pz.to_string()),
            ("e2", t0.e2.to_string()),
            ("enstrophy_x4", ens0.to_string()),
            ("sha256", hex_digest(&base)),
        ],
    );
    let mut off = base.clone();
    let mut on = base.clone();
    let mut s1 = vec![0u8; SLOTS];
    let mut s2 = vec![0u8; SLOTS];
    let mut injected = 0i64;
    let (mut first64, mut last64) = (0i64, 0i64);
    let mut z_tick: Option<usize> = None;
    let mut local32_tick: Option<usize> = None;
    let mut c_ok = true;
    let mut front_hist = Vec::new();
    let mut ens_off_end = ens0;
    let mut ens_on_end = ens0;
    let base_plane32 = plane_hash(&base, N / 2);
    let mut inj_series = Vec::with_capacity(ticks);
    let mut snaps = vec![snap(0, &base, &base)];
    for t in 0..ticks {
        let inj = solar_wind(&mut on);
        injected += inj;
        inj_series.push(inj);
        if t < 64 {
            first64 += inj;
        }
        if t >= ticks.saturating_sub(64) {
            last64 += inj;
        }
        tick(&mut off, &mut s1, gates, t);
        tick(&mut on, &mut s2, gates, t);
        let now = t + 1;
        let a = totals(&off);
        let b = totals(&on);
        assert_eq!(a, t0, "wind-off conservation at t={now}");
        assert_eq!(
            b,
            Totals {
                px: t0.px + injected,
                ..t0
            },
            "wind-on: mass, 2E, py, pz exact; px = px0 + injected, t={now}"
        );
        if now == 32 || now == 192 || now == ticks {
            snaps.push(snap(now, &off, &on));
        }
        let fd = front_distance(&off, &on);
        if let Some(d) = fd {
            if d > now {
                c_ok = false;
            }
        }
        if now <= N / 2 + 1 {
            front_hist.push(fd.map_or(-1, |d| d as i64));
        }
        if z_tick.is_none() && digest(&on) != digest(&off) {
            z_tick = Some(now);
        }
        if local32_tick.is_none() && plane_hash(&on, N / 2) != plane_hash(&off, N / 2) {
            local32_tick = Some(now);
        }
        let keep = now % 8 == 0 || now <= 4 || now == ticks;
        if keep {
            let ja = momentum(&off);
            let jb = momentum(&on);
            let (ea, eb) = (enstrophy(&ja), enstrophy(&jb));
            ens_off_end = ea;
            ens_on_end = eb;
            let m = modes(&on);
            let q: i64 = m.iter().sum();
            series.row(
                "TICK",
                &[
                    ("lattice", name.into()),
                    ("t", now.to_string()),
                    ("injected_px_this_tick", inj.to_string()),
                    ("injected_px_total", injected.to_string()),
                    ("front", fd.map_or("none".into(), |d| d.to_string())),
                    ("enstrophy_x4_off", ea.to_string()),
                    ("enstrophy_x4_on", eb.to_string()),
                    ("kept_slots_off", same_slots(&off, gguf_lattice).to_string()),
                    ("kept_slots_on", same_slots(&on, gguf_lattice).to_string()),
                    ("outer_Q", q.to_string()),
                    (
                        "outer_modes",
                        m.iter().map(i64::to_string).collect::<Vec<_>>().join(","),
                    ),
                ],
            );
        }
    }
    // where the injected momentum sits at the end: on - off, per x-plane
    let pa = px_by_plane(&momentum(&off));
    let pb = px_by_plane(&momentum(&on));
    let delta: Vec<i64> = pa.iter().zip(pb.iter()).map(|(a, b)| b - a).collect();
    let total_delta: i64 = delta.iter().sum();
    assert_eq!(total_delta, injected, "all injected px is in the lattice");
    let near: i64 = (0..N)
        .filter(|&x| x.min(N - x) <= 8)
        .map(|x| delta[x])
        .sum();
    let carried = total_delta - near;
    let carried_pm = if total_delta == 0 {
        0
    } else {
        carried * 1000 / total_delta
    };
    rec.row(
        "WIND",
        &[
            ("lattice", name.into()),
            ("ticks", ticks.to_string()),
            ("injected_px_total", injected.to_string()),
            ("injected_first64", first64.to_string()),
            ("injected_last64", last64.to_string()),
            ("px_on_minus_off_within_8_planes", near.to_string()),
            ("px_on_minus_off_beyond_8_planes", carried.to_string()),
            ("carried_beyond_8_permille", carried_pm.to_string()),
            (
                "delta_px_by_plane",
                delta
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ],
    );
    rec.row(
        "Z_AND_C",
        &[
            ("lattice", name.into()),
            (
                "global_sha256_differs_first_at_t",
                z_tick.map_or("never".into(), |t| t.to_string()),
            ),
            (
                "plane_x32_hash_differs_first_at_t",
                local32_tick.map_or("never".into(), |t| t.to_string()),
            ),
            ("plane_x32_hash_at_t0", hex(&base_plane32)[..16].to_string()),
            ("front_never_exceeds_t", u8::from(c_ok).to_string()),
            (
                "front_t1_to_t33",
                front_hist
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ],
    );
    RunSummary {
        name: name.into(),
        injected_first64: first64,
        injected_last64: last64,
        injected_total: injected,
        carried_past8_permille: carried_pm,
        ens_off_end,
        ens_on_end,
        ens_start: ens0,
        inj_series,
        snaps,
    }
}

fn reverse_check(start: &[u8], gates: &[Gate], steps: usize) -> (bool, u64) {
    let mut cur = start.to_vec();
    let mut s = vec![0u8; SLOTS];
    for t in 0..steps {
        tick(&mut cur, &mut s, gates, t);
    }
    let moved = (SLOTS as u64) - same_slots(&cur, start);
    for t in (0..steps).rev() {
        unstream(&cur, &mut s);
        std::mem::swap(&mut cur, &mut s);
        uncollide(&mut cur, gates, t);
    }
    (cur == start, moved)
}

/// The figure: the 27 GGUF channels (z = 32 layer), the wind-on density and the (on - off)
/// x-momentum for the Asolaria lattice at four ticks, and the injection per tick for all three.
fn render(lattice: &[u8], sums: &[RunSummary]) -> Vec<u8> {
    use mixed_gguf_net::png::{blue, diverging, Canvas};
    let bg = [2u8, 4, 12];
    let (w, h) = (1092usize, 1016usize);
    let mut c = Canvas::new(w, h, bg);
    // row A: 27 channels, 9 x 3 tiles
    let tile = 64usize;
    let ax = (w - (9 * tile + 8 * 8)) / 2;
    for k in 0..K {
        let (col, row) = (k % 9, k / 9);
        let (x0, y0) = (ax + col * (tile + 8), 16 + row * (tile + 8));
        if k == REST {
            c.rect(x0 - 2, y0 - 2, tile + 4, tile + 4, [255, 215, 94]);
        }
        for y in 0..N {
            for x in 0..N {
                let t = lattice[mixed_gguf_net::lattice::idx(x, y, N / 2) * K + k];
                let col = match t {
                    0 => [10u8, 16, 34],
                    1 => [31, 111, 224],
                    _ => [235, 245, 255],
                };
                c.set(x0 + x, y0 + y, col);
            }
        }
    }
    let snaps = &sums[0].snaps;
    let dmax = snaps
        .iter()
        .flat_map(|s| s.density.iter())
        .copied()
        .max()
        .unwrap_or(1);
    let pmax = snaps
        .iter()
        .flat_map(|s| s.dpx.iter())
        .map(|v| v.abs())
        .max()
        .unwrap_or(1);
    let (bx, by) = (16usize, 16 + 3 * tile + 2 * 8 + 16);
    for (i, sn) in snaps.iter().enumerate().take(4) {
        let x0 = bx + i * (256 + 12);
        for y in 0..N {
            for x in 0..N {
                let v = sn.density[y * N + x];
                let d = sn.dpx[y * N + x];
                c.rect(x0 + x * 4, by + y * 4, 4, 4, blue(v, dmax));
                c.rect(x0 + x * 4, by + 268 + y * 4, 4, 4, diverging(d, pmax));
            }
        }
        // tick marker: a gold bar whose length is t / ticks of the tile width
        let ticks = sums[0].inj_series.len().max(1);
        c.rect(x0, by - 6, (256 * sn.t) / ticks, 3, [255, 215, 94]);
    }
    // row D: injection per tick
    let (dx, dy, dw, dh) = (16usize, by + 268 + 256 + 16, w - 32, 220usize);
    c.rect(dx, dy, dw, 1, [31, 58, 102]);
    c.rect(dx, dy + dh, dw, 1, [31, 58, 102]);
    let imax = sums
        .iter()
        .flat_map(|s| s.inj_series.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let colours = [[255u8, 215, 94], [95, 168, 255], [150, 150, 150]];
    for (s, col) in sums.iter().zip(colours.iter()).rev() {
        let n = s.inj_series.len().max(2) as i64;
        let pt = |i: usize, v: i64| {
            (
                dx as i64 + (i as i64) * (dw as i64 - 1) / (n - 1),
                (dy + dh) as i64 - v * (dh as i64 - 4) / imax,
            )
        };
        for i in 1..s.inj_series.len() {
            let (x0, y0) = pt(i - 1, s.inj_series[i - 1]);
            let (x1, y1) = pt(i, s.inj_series[i]);
            c.line(x0, y0, x1, y1, *col);
        }
    }
    c.png()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args[1] != "all" {
        eprintln!("usage: mixed-gguf-net all DATA_DIR OUT_DIR [TICKS]");
        std::process::exit(2);
    }
    let data = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    let ticks: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(576);
    fs::create_dir_all(out).expect("out dir");
    let mut files = Vec::new();

    // 1. lattice
    let (lattice, channels) = build(data);
    print!("{}", channels.text());
    write(out, "CHANNELS.hbp", &channels.text(), &mut files);
    fs::write(out.join("lattice.trits"), &lattice).expect("lattice");
    files.push(("lattice.trits".into(), lattice.len(), hex_digest(&lattice)));

    // 2. gates
    let gates = make_gates(true);
    let mut rec = Hbp::new();
    let fixed40 = gates.iter().filter(|g| g.lut[40] == 40).count();
    println!(
        "{}",
        rec.row(
            "GATES",
            &[
                ("count", gates.len().to_string()),
                ("involutive_and_conserving", "1".into()),
                ("fix_code40", fixed40.to_string()),
                ("per_cell_per_tick", "32".into()),
                ("schedule_period_ticks", "9".into()),
            ],
        )
    );

    // 3. controls: reversibility and null
    let mut tg = lattice.clone();
    imprint_taylor_green(&mut tg);
    let (rev_ok, moved) = reverse_check(&tg, &gates, 72);
    println!(
        "{}",
        rec.row(
            "REVERSE",
            &[
                ("lattice", "asolaria_taylor_green".into()),
                ("steps", "72".into()),
                ("slots_moved_at_t72", moved.to_string()),
                ("every_trit_restored", u8::from(rev_ok).to_string()),
            ],
        )
    );
    let uni = vec![1u8; SLOTS];
    let mut u = uni.clone();
    let mut s = vec![0u8; SLOTS];
    for t in 0..72 {
        tick(&mut u, &mut s, &gates, t);
    }
    println!(
        "{}",
        rec.row(
            "NULL",
            &[("uniform_unchanged_after_72", u8::from(u == uni).to_string())]
        )
    );
    let mut fs_state = tg.clone();
    for _ in 0..N {
        stream(&fs_state, &mut s);
        std::mem::swap(&mut fs_state, &mut s);
    }
    let free_period = fs_state == tg;
    println!(
        "{}",
        rec.row(
            "FREE_STREAM",
            &[(
                "returns_to_start_after_64",
                u8::from(free_period).to_string()
            )]
        )
    );

    // 4. the three lattices, wind off and on in lockstep
    let mut series = Hbp::new();
    let shuf = shuffled(&lattice);
    let mut sums = Vec::new();
    for (name, lat) in [
        ("asolaria", &lattice),
        ("shuffled", &shuf),
        ("uniform", &uni),
    ] {
        let n0 = rec.len();
        sums.push(run_lattice(name, lat, &gates, ticks, &mut rec, &mut series));
        for line in rec.text().lines().skip(n0) {
            println!("{line}");
        }
    }
    for sm in &sums {
        println!(
            "{}",
            rec.row(
                "ABSORB",
                &[
                    ("lattice", sm.name.clone()),
                    ("injected_first64", sm.injected_first64.to_string()),
                    ("injected_last64", sm.injected_last64.to_string()),
                    (
                        "declined",
                        u8::from(sm.injected_last64 < sm.injected_first64).to_string()
                    ),
                    ("injected_total", sm.injected_total.to_string()),
                    (
                        "carried_beyond_8_permille",
                        sm.carried_past8_permille.to_string()
                    ),
                    ("enstrophy_x4_start", sm.ens_start.to_string()),
                    ("enstrophy_x4_end_off", sm.ens_off_end.to_string()),
                    ("enstrophy_x4_end_on", sm.ens_on_end.to_string()),
                ],
            )
        );
    }
    let (a, b) = (sums[0].injected_total, sums[1].injected_total);
    let diff_pm = if a == 0 { 0 } else { (a - b).abs() * 1000 / a };
    println!(
        "{}",
        rec.row(
            "CONTENT",
            &[
                ("asolaria_injected", a.to_string()),
                ("shuffled_injected", b.to_string()),
                ("difference_permille", diff_pm.to_string()),
                ("within_5pct", u8::from(diff_pm <= 50).to_string()),
            ],
        )
    );

    // 5. the jump to t = 10^13 ticks, from what is exact
    let t0 = totals(&tg);
    let px_bound: i64 = 2 * 9 * CELLS as i64;
    let d_sum: i64 = (0..27)
        .fold((0i64, 1i64), |(s, p), _| ((s + p) % P_MOD, p * 9 % P_MOD))
        .0;
    let q0 = t0.mass % P_MOD;
    let t_mod = TEN_TRILLION % P_MOD;
    let q_jump = (q0 - t_mod * d_sum % P_MOD).rem_euclid(P_MOD);
    println!(
        "{}",
        rec.row(
            "JUMP",
            &[
                ("t", TEN_TRILLION.to_string()),
                ("mass", t0.mass.to_string()),
                ("e2", t0.e2.to_string()),
                ("py", t0.py.to_string()),
                ("pz", t0.pz.to_string()),
                ("px_wind_off", t0.px.to_string()),
                (
                    "free_stream_state_equals_start",
                    u8::from(TEN_TRILLION % (N as i64) == 0 && free_period).to_string()
                ),
                ("px_upper_bound", px_bound.to_string()),
                (
                    "mean_injection_per_tick_upper_bound_num",
                    (px_bound - t0.px).to_string()
                ),
                (
                    "mean_injection_per_tick_upper_bound_den",
                    TEN_TRILLION.to_string()
                ),
                ("outer_D_mod_P", d_sum.to_string()),
                ("outer_Q_mod_P_at_t_nu1", q_jump.to_string()),
                ("gated_microstate_at_t", "not_jumpable_must_be_run".into()),
            ],
        )
    );

    let fig = render(&lattice, &sums);
    fs::write(out.join("figure.png"), &fig).expect("figure");
    files.push(("figure.png".into(), fig.len(), hex_digest(&fig)));
    write(out, "RESULT.hbp", &rec.text(), &mut files);
    write(out, "SERIES.hbp", &series.text(), &mut files);
    let idx = index("RESULTINDEX", &files);
    fs::write(out.join("INDEX.hbi"), &idx).expect("index");
    let mut sums_txt = String::new();
    for (f, _, h) in &files {
        sums_txt.push_str(&format!("{h}  {f}\n"));
    }
    sums_txt.push_str(&format!("{}  INDEX.hbi\n", hex_digest(idx.as_bytes())));
    fs::write(out.join("SHA256SUMS"), sums_txt).expect("sums");
    println!("DONE|files={}|json=0", files.len() + 1);
}
