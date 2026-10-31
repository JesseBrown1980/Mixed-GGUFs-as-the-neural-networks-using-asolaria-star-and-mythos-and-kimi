//! `virtual`: the balanced-trit virtual layer over the three lattices.
//! Replays the committed run (same dynamics, checked against results/SERIES.hbp as it
//! goes) and at every checkpoint records the Gram matrices that make every quadratic
//! readout of every virtual combination exact.

use crate::{build, shuffled, tick};
use mixed_gguf_net::hbp::{index, Hbp};
use mixed_gguf_net::lattice::make_gates;
use mixed_gguf_net::physics::{curl, enstrophy, imprint_taylor_green, momentum, solar_wind, SLOTS};
use mixed_gguf_net::png::Canvas;
use mixed_gguf_net::sha256::hex_digest;
use mixed_gguf_net::virtual_layer::{
    corr_permille, diag_part, gram, linear, overhead, quad, sign_classes, weights, Gram, B,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const NAMES: [&str; B] = ["asolaria", "shuffled", "uniform"];

fn get(line: &str, key: &str) -> Option<String> {
    line.split('|')
        .find_map(|kv| kv.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
        .map(str::to_string)
}

/// (lattice, t) -> (intake this tick, enstrophy off, enstrophy on) from the committed series.
fn committed_series(results: &Path) -> BTreeMap<(String, usize), (i64, i64, i64)> {
    let text = fs::read_to_string(results.join("SERIES.hbp")).expect("committed SERIES.hbp");
    let mut m = BTreeMap::new();
    for l in text.lines().filter(|l| l.starts_with("TICK|")) {
        let num = |k: &str| get(l, k).and_then(|v| v.parse::<i64>().ok()).expect(k);
        let t = get(l, "t")
            .and_then(|v| v.parse::<usize>().ok())
            .expect("t");
        m.insert(
            (get(l, "lattice").expect("lattice"), t),
            (
                num("injected_px_this_tick"),
                num("enstrophy_x4_off"),
                num("enstrophy_x4_on"),
            ),
        );
    }
    m
}

fn committed_totals(results: &Path) -> [i64; B] {
    let text = fs::read_to_string(results.join("RESULT.hbp")).expect("committed RESULT.hbp");
    let mut out = [0i64; B];
    for l in text.lines().filter(|l| l.starts_with("WIND|")) {
        let name = get(l, "lattice").expect("lattice");
        let v: i64 = get(l, "injected_px_total")
            .and_then(|v| v.parse().ok())
            .expect("total");
        if let Some(i) = NAMES.iter().position(|n| *n == name) {
            out[i] = v;
        }
    }
    out
}

fn tri(g: &Gram) -> String {
    format!(
        "{},{},{},{},{},{}",
        g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2]
    )
}

fn corrs(g: &Gram) -> [i128; 3] {
    [
        corr_permille(g, 0, 1),
        corr_permille(g, 0, 2),
        corr_permille(g, 1, 2),
    ]
}

fn cstr(c: &[i64; B]) -> String {
    c.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
}

struct Checkpoint {
    t: usize,
    gj: [Gram; 3],
    gw: [Gram; 3],
}

#[allow(clippy::too_many_lines)]
pub fn run(data: &Path, committed: &Path, out: &Path, ticks: usize) {
    fs::create_dir_all(out).expect("out");
    let (lattice, _) = build(data);
    let gates = make_gates(true);
    let mut off: Vec<Vec<u8>> = vec![lattice.clone(), shuffled(&lattice), vec![1u8; SLOTS]];
    for f in off.iter_mut() {
        imprint_taylor_green(f);
    }
    let mut on = off.clone();
    let mut scratch = vec![0u8; SLOTS];
    let series = committed_series(committed);
    let totals_committed = committed_totals(committed);
    let mut intake = [0i64; B];
    let mut intake_series: Vec<[i64; B]> = Vec::with_capacity(ticks);
    let mut checks: Vec<Checkpoint> = Vec::new();
    let mut matched_rows = 0usize;
    let mut nonneg = true;

    let checkpoint = |t: usize, off: &[Vec<u8>], on: &[Vec<u8>], nonneg: &mut bool| {
        let joff: Vec<Vec<[i64; 3]>> = off.iter().map(|f| momentum(f)).collect();
        let jon: Vec<Vec<[i64; 3]>> = on.iter().map(|f| momentum(f)).collect();
        let jresp: Vec<Vec<[i64; 3]>> = jon
            .iter()
            .zip(joff.iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
                    .collect()
            })
            .collect();
        let woff: Vec<Vec<[i64; 3]>> = joff.iter().map(|j| curl(j)).collect();
        let won: Vec<Vec<[i64; 3]>> = jon.iter().map(|j| curl(j)).collect();
        let wresp: Vec<Vec<[i64; 3]>> = jresp.iter().map(|j| curl(j)).collect();
        let g3 = |f: &[Vec<[i64; 3]>]| gram([&f[0], &f[1], &f[2]]);
        let cp = Checkpoint {
            t,
            gj: [g3(&joff), g3(&jon), g3(&jresp)],
            gw: [g3(&woff), g3(&won), g3(&wresp)],
        };
        // V2: Gram diagonals are the branch enstrophies
        for i in 0..B {
            assert_eq!(cp.gw[0][i][i], i128::from(enstrophy(&joff[i])), "diag off");
            assert_eq!(cp.gw[1][i][i], i128::from(enstrophy(&jon[i])), "diag on");
        }
        // V3: every virtual quadratic readout is non-negative
        for c in weights() {
            for g in cp.gj.iter().chain(cp.gw.iter()) {
                *nonneg &= quad(&c, g) >= 0;
            }
        }
        cp
    };

    checks.push(checkpoint(0, &off, &on, &mut nonneg));
    for t in 0..ticks {
        let mut inj = [0i64; B];
        for i in 0..B {
            inj[i] = solar_wind(&mut on[i]);
            intake[i] += inj[i];
        }
        intake_series.push(inj);
        for i in 0..B {
            tick(&mut off[i], &mut scratch, &gates, t);
            tick(&mut on[i], &mut scratch, &gates, t);
        }
        let now = t + 1;
        let committed_row = now % 8 == 0 || now <= 4 || now == ticks;
        if committed_row || now % 8 == 0 {
            let cp = checkpoint(now, &off, &on, &mut nonneg);
            for i in 0..B {
                let key = (NAMES[i].to_string(), now);
                let (ci, eoff, eon) = *series.get(&key).expect("committed series row");
                assert_eq!(inj[i], ci, "intake {} t={now}", NAMES[i]);
                assert_eq!(
                    cp.gw[0][i][i],
                    i128::from(eoff),
                    "enstrophy off {} t={now}",
                    NAMES[i]
                );
                assert_eq!(
                    cp.gw[1][i][i],
                    i128::from(eon),
                    "enstrophy on {} t={now}",
                    NAMES[i]
                );
                matched_rows += 1;
            }
            if now % 8 == 0 || now == ticks {
                checks.push(cp);
            }
        }
    }
    let committed_ticks = fs::read_to_string(committed.join("RESULT.hbp"))
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("WIND|"))
                .and_then(|l| get(l, "ticks"))
        })
        .and_then(|v| v.parse::<usize>().ok());
    let totals_compared = committed_ticks == Some(ticks);
    if totals_compared {
        assert_eq!(
            intake, totals_committed,
            "intake totals match committed RESULT.hbp"
        );
    }

    let mut rec = Hbp::new();
    let mut vs = Hbp::new();
    rec.row(
        "VIRTUAL",
        &[
            ("branches", NAMES.join(",")),
            ("weights", "c_in_-1_0_+1_cubed".into()),
            ("nonzero_vectors", weights().len().to_string()),
            ("sign_classes", sign_classes().len().to_string()),
            ("ticks", ticks.to_string()),
            ("checkpoints", checks.len().to_string()),
            ("committed_rows_matched", matched_rows.to_string()),
            ("branch_intake_totals", cstr(&intake)),
            (
                "equal_to_committed_totals",
                u8::from(totals_compared).to_string(),
            ),
        ],
    );
    for cp in &checks {
        let names = ["off", "on", "resp"];
        let mut fields: Vec<(&str, String)> = vec![("t", cp.t.to_string())];
        let mut owned: Vec<(String, String)> = Vec::new();
        for (k, nm) in names.iter().enumerate() {
            owned.push((format!("GJ_{nm}"), tri(&cp.gj[k])));
            owned.push((format!("GW_{nm}"), tri(&cp.gw[k])));
            let cj = corrs(&cp.gj[k]);
            let cw = corrs(&cp.gw[k]);
            owned.push((
                format!("corrJ_{nm}_AS_AU_SU"),
                format!("{},{},{}", cj[0], cj[1], cj[2]),
            ));
            owned.push((
                format!("corrW_{nm}_AS_AU_SU"),
                format!("{},{},{}", cw[0], cw[1], cw[2]),
            ));
        }
        for (k, v) in &owned {
            fields.push((k.as_str(), v.clone()));
        }
        vs.row("GRAM", &fields);
    }

    // V1: linearity of the intake readout, all 26 weights
    let mut v1 = true;
    for c in weights() {
        let per_tick: i64 = intake_series.iter().map(|v| linear(&c, v)).sum();
        v1 &= per_tick == linear(&c, &intake);
    }
    let last = checks.last().expect("checkpoint");
    // the full table at the final tick, one row per sign class
    for c in sign_classes() {
        rec.row(
            "WEIGHT",
            &[
                ("c", cstr(&c)),
                ("overhead", overhead(&c).to_string()),
                ("balanced", u8::from(c.iter().sum::<i64>() == 0).to_string()),
                ("virtual_intake_total", linear(&c, &intake).to_string()),
                ("flow_energy_off", quad(&c, &last.gj[0]).to_string()),
                ("flow_energy_on", quad(&c, &last.gj[1]).to_string()),
                ("flow_energy_resp", quad(&c, &last.gj[2]).to_string()),
                ("enstrophy_off", quad(&c, &last.gw[0]).to_string()),
                ("enstrophy_on", quad(&c, &last.gw[1]).to_string()),
                ("enstrophy_resp", quad(&c, &last.gw[2]).to_string()),
                (
                    "enstrophy_on_vs_uncorrelated_permille",
                    (quad(&c, &last.gw[1]) * 1000 / diag_part(&c, &last.gw[1]).max(1)).to_string(),
                ),
            ],
        );
    }
    // T1: balanced weight that cancels the wind-on vortex field best
    let t1 = sign_classes()
        .into_iter()
        .filter(|c| c.iter().sum::<i64>() == 0)
        .map(|c| {
            (
                quad(&c, &last.gw[1]) * 1000 / diag_part(&c, &last.gw[1]).max(1),
                c,
            )
        })
        .min()
        .expect("balanced");
    // T2: coherence of the three wind responses (vortex field)
    let all = [1i64, 1, 1];
    let t2 = quad(&all, &last.gw[2]) * 1000 / diag_part(&all, &last.gw[2]).max(1);
    let t2j = quad(&all, &last.gj[2]) * 1000 / diag_part(&all, &last.gj[2]).max(1);
    // T3: virtual ground state of the wind-on flow
    let t3 = weights()
        .into_iter()
        .map(|c| (quad(&c, &last.gj[1]), c))
        .min()
        .expect("weights");
    let single_min = (0..B).map(|i| last.gj[1][i][i]).min().unwrap_or(0);
    rec.row(
        "TASK",
        &[
            ("id", "T1".into()),
            ("analog", "error_mitigation".into()),
            ("best_balanced_c", cstr(&t1.1)),
            (
                "wind_on_enstrophy_vs_uncorrelated_permille",
                t1.0.to_string(),
            ),
        ],
    );
    rec.row(
        "TASK",
        &[
            ("id", "T2".into()),
            ("analog", "classification".into()),
            ("c", "1,1,1".into()),
            ("wind_response_vortex_coherence_permille", t2.to_string()),
            ("wind_response_momentum_coherence_permille", t2j.to_string()),
            ("note", "1000_uncorrelated_3000_identical".into()),
        ],
    );
    rec.row(
        "TASK",
        &[
            ("id", "T3".into()),
            ("analog", "ground_state".into()),
            ("minimiser_c", cstr(&t3.1)),
            ("virtual_flow_energy_on", t3.0.to_string()),
            ("smallest_single_branch", single_min.to_string()),
            (
                "balanced",
                u8::from(t3.1.iter().sum::<i64>() == 0).to_string(),
            ),
        ],
    );
    let cj = corrs(&last.gj[2]);
    let cw_off = corrs(&last.gw[0]);
    let verdicts = [
        ("V1", v1),
        ("V2", true),
        ("V3", nonneg),
        ("V4", cj.iter().all(|&x| x >= 900)),
        ("V5", cw_off[0] < 500),
        ("V6", t3.1.iter().sum::<i64>() == 0),
    ];
    let measured = [
        "virtual_intake_per_tick_sums_equal_linear_totals_all_26".to_string(),
        format!("gram_diagonals_equal_committed_enstrophies_at_{matched_rows}_rows"),
        "every_cT_G_c_ge_0_all_26_all_checkpoints_all_6_grams".to_string(),
        format!(
            "resp_momentum_corr_permille_AS_AU_SU={},{},{}",
            cj[0], cj[1], cj[2]
        ),
        format!("wind_off_curl_corr_permille_AS={}", cw_off[0]),
        format!("T3_minimiser={}", cstr(&t3.1)),
    ];
    for ((id, pass), m) in verdicts.iter().zip(measured.iter()) {
        println!(
            "{}",
            rec.row(
                "VERDICT",
                &[
                    ("id", (*id).into()),
                    ("measured", m.clone()),
                    ("pass", u8::from(*pass).to_string())
                ],
            )
        );
    }

    // figure: wind-on vortex correlation over time (AS gold, AU blue, SU grey) on top,
    // wind-response momentum correlation below; y from -1000 to +1000 per-mille.
    let (w, h) = (1092usize, 560usize);
    let mut c = Canvas::new(w, h, [2, 4, 12]);
    let cols = [[255u8, 215, 94], [95, 168, 255], [150, 150, 150]];
    for (panel, field) in [(0usize, 1usize), (1, 2)] {
        let (x0, y0, pw, ph) = (16usize, 16 + panel * 272, w - 32, 256usize);
        c.rect(x0, y0, pw, 1, [31, 58, 102]);
        c.rect(x0, y0 + ph / 2, pw, 1, [31, 58, 102]);
        c.rect(x0, y0 + ph, pw, 1, [31, 58, 102]);
        let n = checks.len().max(2) as i64;
        for (pair, col) in cols.iter().enumerate() {
            let pt = |i: usize| {
                let g = if panel == 0 {
                    &checks[i].gw[field]
                } else {
                    &checks[i].gj[field]
                };
                let v = corrs(g)[pair] as i64;
                (
                    x0 as i64 + i as i64 * (pw as i64 - 1) / (n - 1),
                    (y0 + ph / 2) as i64 - v * (ph as i64 / 2 - 2) / 1000,
                )
            };
            for i in 1..checks.len() {
                let (a, b) = pt(i - 1);
                let (x, y) = pt(i);
                c.line(a, b, x, y, *col);
            }
        }
    }
    let fig = c.png();

    let mut files = Vec::new();
    for (name, body) in [
        ("VIRTUAL.hbp", rec.text()),
        ("VIRTUAL-SERIES.hbp", vs.text()),
    ] {
        fs::write(out.join(name), &body).expect("write");
        files.push((name.to_string(), body.len(), hex_digest(body.as_bytes())));
    }
    fs::write(out.join("virtual-figure.png"), &fig).expect("fig");
    files.push(("virtual-figure.png".into(), fig.len(), hex_digest(&fig)));
    let idx = index("VIRTUALINDEX", &files);
    fs::write(out.join("INDEX.hbi"), &idx).expect("index");
    let mut sums = String::new();
    for (f, _, h) in &files {
        sums.push_str(&format!("{h}  {f}\n"));
    }
    sums.push_str(&format!("{}  INDEX.hbi\n", hex_digest(idx.as_bytes())));
    fs::write(out.join("SHA256SUMS"), sums).expect("sums");
    for l in rec
        .text()
        .lines()
        .filter(|l| l.starts_with("TASK") || l.starts_with("WEIGHT"))
    {
        println!("{l}");
    }
    println!("DONE|files={}|json=0", files.len() + 1);
}
