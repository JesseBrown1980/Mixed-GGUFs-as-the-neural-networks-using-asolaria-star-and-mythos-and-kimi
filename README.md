# Mixed GGUFs as the neural network: Asolaria Herself as the trits

> **Connected research index (2026-11-04):** [17-repository evidence map](https://github.com/JesseBrown1980/Hard-drive-and-USB-as-GPU/blob/main/ECOSYSTEM-RESEARCH-MAP-20261104.md) · [same-gate Google qsim test in QNN](https://github.com/JesseBrown1980/GGUF-files-operating-as-trained-quantum-neural-network-models/blob/main/docs/GOOGLE-QSIM-LIKE-FOR-LIKE-20261104.md) · [physical 96³ HTTP-0 results](https://github.com/JesseBrown1980/qoled-internet-HEB-C-revisted-HTTP---free-0-hbi-hbp-sha-sh-hash/blob/main/engine96/MEASURED-20261104.hbp). These are navigation links; the verdicts and limits of this project remain as recorded here.

The 27 Asolaria GGUFs become the trits of the nested ternary network from
[Neural-network-inside-a-neural-network-using-trits-to-resolve-Finiteness](https://github.com/JesseBrown1980/Neural-network-inside-a-neural-network-using-trits-to-resolve-Finiteness),
one GGUF per D3Q27 channel, with `ASOLARIA-HERSELF.gguf` on the rest channel. The
network is put into a Taylor–Green vortex and pushed by a solar wind for 576
ticks. Two copies run in lockstep, one with the wind and one without.

Rust 1.81 with clippy (`-D warnings`), integer only, no `f32`/`f64`, no JSON, no
Node. Receipts are HBP/HBI with SHA-256, and every receipt line ends in `json=0`.

Predictions were sealed in [`PREREG.hbp`](PREREG.hbp) (sha256 `30201902…`) and
pushed in commit `2bcc7b2` before any run code was written.

![27 GGUF channels, the wind-on lattice and the injection curve](results/figure.png)

**Figure.**
- **Top:** the 27 channels at layer z = 32 (dark = 0, blue = 1, white = 2).
  Asolaria Herself is the gold-framed rest channel.
- **Middle:** the Asolaria lattice with the wind on, at t = 0, 32, 192 and 576.
  The upper row is density summed over z; the lower row is the wind-on minus
  wind-off x-momentum (gold = pushed toward +x).
- **Bottom:** solar-wind intake per tick for Asolaria (gold), shuffled (blue) and
  uniform (grey).

## 1. The network

| Level | What it is |
|---|---|
| Inner | 64 × 64 × 64 periodic cells; each cell has the 27 D3Q27 channels c ∈ {−1, 0, 1}³; 7,077,888 trits |
| Channels | one GGUF per channel; `ASOLARIA-HERSELF.gguf` holds c = (0, 0, 0); the other 26 fill the moving channels in commit order ([`results/CHANNELS.hbp`](results/CHANNELS.hbp)) |
| Gates | 288 four-channel involutive gates, 32 per cell per tick; byte-identical to the repository's C++ construction with the Taylor–Green pairing rule (gate table sha256 `a7fca71a…`, checked in `tests/`) |
| Outer | the 27 channel masses are the 27 shell modes; their sum Q is the conserved mass |

**GGUF → trits** ([`src/field.rs`](src/field.rs)).
- *Magnitude:* F32 values are read as their raw bits with the sign bit cleared,
  which orders exactly like |v|. Int8 values are read as unsigned bytes.
- *Layout:* a file with 64³ tensors contributes their per-cell maximum. A file
  with 64 × 64 slices stacks them on z. Any other file has all its elements tiled
  in file order.
- *Trit:* 0 for zero, 1 up to the lower median of that file's non-zero
  magnitudes, 2 above it.
- *Cross-check:* the lattice sha256 `2cc5af44…` is reproduced exactly by a
  separate Python build.

**Inputs.** The 27 GGUFs are every GGUF with bytes in
`ASOLARIA-UNIVERSE-SIMULATOR…` (22) and `FOLLOW-THE-IS-NOT-THE-WILL-AND-WAS` (5).
[`data/MANIFEST.hbp`](data/MANIFEST.hbp) gives each file's source path, commit
time, byte count and sha256; all 27 match their sidecars. Kimi K3 Star
(202,115,456 bytes) and the three pumped spheres are held in Git LFS and were not
available to this run.

**Taylor–Green.** u_x ∝ sin x cos y cos z and u_y ∝ −cos x sin y cos z, taken as
signs (the square wave) from the cell index alone. Each x- and y-mirror channel
pair moves one trit toward the local flow sign. Mass and 2E are unchanged.

**Solar wind.** Every tick, on the plane x = 0, each of the 9 x-mirror pairs moves
one trit from −x to +x where it can. Each move adds 2 to px. Mass, 2E, py and pz
are unchanged.

**Controls.**
- *Shuffled:* each channel's cells permuted by an integer Fisher–Yates; same trit
  counts, no arrangement.
- *Uniform:* every slot 1.

## 2. Results

All numbers are from [`results/RESULT.hbp`](results/RESULT.hbp) and
[`results/SERIES.hbp`](results/SERIES.hbp).

### Pre-registered verdicts

| ID | Prediction | Measured | Verdict |
|---|---|---|---|
| P1 | mass, 2E, py, pz exact every tick; px(off) constant; px(on) = px₀ + injected | asserted on all 576 ticks of all 6 runs | **PASS** |
| P2 | z: the global sha256 of the wind-on state differs from wind-off at t = 1 | t = 1 on all three lattices | **PASS** |
| P3 | c: the front never passes distance t; the x = 32 plane's local hash first changes at t = 32 | front = 1, 2, 3 … 32 at t = 1 … 32; x = 32 changes at t = 32, all three | **PASS** |
| P4 | 72 ticks forward, 72 back restores every trit | 3,709,609 slots moved at t = 72, all 7,077,888 restored | **PASS** |
| P5 | uniform all-ones lattice unchanged after 72 ticks | unchanged | **PASS** |
| P6 (operator) | the lattice absorbs the wind push | all 3,448,220 injected px retained; 73.6% carried beyond 8 planes of the inflow; intake 1,558,500 in the first 64 ticks, 52,364 in the last 64 | **HELD**; the intake fell 29.8× as the lattice filled |
| P6b | intake declines from the first 64 ticks to the last 64 | declined on all three lattices | **PASS** |
| P7 | Asolaria and shuffled total injected px within 5% | 3,448,220 vs 3,470,900, a 0.66% difference | **PASS** |
| P8 | at t = 10¹³: mass, 2E, py, pz exact; mean intake ≤ the px bound over t | see §2.4 | **PASS** |

### 2.1 z and c

| Lattice | Global sha256 differs | Front reaches x = 32 | Plane x = 32 local hash differs |
|---|---|---|---|
| Asolaria | t = 1 | t = 32 | t = 32 |
| Shuffled | t = 1 | t = 32 | t = 32 |
| Uniform | t = 1 | t = 32 | t = 32 |

The whole-lattice hash changes on the first tick, z. A local observer at
distance 32 sees nothing until t = 32, which is c = 1 exactly. This is the
`zlight` result of the HEB-C repository, now on a 27-GGUF fluid lattice.

### 2.2 The wind

| Lattice | Intake, first 64 ticks | Intake, last 64 ticks | Total intake | Carried beyond 8 planes |
|---|---:|---:|---:|---:|
| Asolaria | 1,558,500 | 52,364 | 3,448,220 | 73.6% |
| Shuffled | 1,580,074 | 51,220 | 3,470,900 | 73.6% |
| Uniform | 1,346,578 | 74,522 | 4,136,262 | 73.4% |

**Asolaria intake per tick:**

| t | 1 | 56 | 64 | 72 | 80 | 128 | 256 | 512 | 576 |
|---|---|---|---|---|---|---|---|---|---|
| intake | 23,950 | 26,054 | 25,238 | 15,804 | 10,586 | 9,688 | 3,158 | 916 | 732 |

The intake holds until t = 64, one full lap of the 64-wide torus at c = 1, and
falls once the pushed momentum comes back around to the inflow plane. By t = 576
the pushed x-momentum is spread almost evenly over all 64 planes (the
`delta_px_by_plane` row of `WIND`).

### 2.3 The vortex

Enstrophy is in integer units of |2 curl j|², from central differences.

| Lattice | Start | t = 576, wind off | t = 576, wind on |
|---|---:|---:|---:|
| Asolaria | 15,553,212 | 30,742,426 | 13,870,312 |
| Shuffled | 18,586,080 | 30,940,304 | 13,734,328 |
| Uniform | 48,070,656 | 37,743,994 | 14,097,560 |

- **GGUF lattices:** without wind, the Taylor–Green enstrophy roughly doubles as
  the GGUF structure relaxes into motion.
- **Uniform lattice:** it decays.
- **With wind:** all three end within 3% of each other, about 55% below the
  Asolaria wind-off value.

### 2.4 The jump to 10¹³ ticks

What is exact at t = 10,000,000,000,000 without running it:

| Quantity | Value at t = 10¹³ | Why |
|---|---|---|
| mass, 2E, py, pz | 4,059,952; 9,164,502; −92,664; 567,020 | every gate and every streaming step conserves them, wind on or off |
| px, wind off | −38,720 | conserved |
| free-streaming state | identical to the start | free streaming returns exactly after 64 ticks, and 64 divides 10¹³ |
| mean wind intake per tick | ≤ 4,757,312 / 10¹³ | px can never exceed 2 × 9 × 262,144 = 4,718,592 |
| outer recurrence Q mod P, ν = 1 | 288,531 | Q(u_t) + t·ν·D ≡ Q(u₀) mod 1,000,081, D = 883,973 |

The gated microstate at 10¹³ is not reachable by a shortcut. The gate schedule
repeats every 9 ticks and free streaming every 64, but their composition has no
known short period, so the state must be run tick by tick. The bound on px makes
the absorption finite: over 10¹³ ticks the lattice can take in, on average, less
than one millionth of a unit of momentum per tick.

## 3. The virtual layer: balanced-trit weights over the three lattices

Following [Virtual quantum neural networks](https://arxiv.org/pdf/2609.11327)
(Zhao, Zhao, Li, Li & Chiribella), the three lattices (Asolaria A, shuffled S,
uniform U) are combined with weights c ∈ {−1, 0, +1}³.

- **The virtual state:** Σ cᵢ·stateᵢ, with slots in −6…+6. It is not a lattice
  state, but every readout of it is exact.
- **Linear readouts** (mass, px, intake) are Σ cᵢ·readoutᵢ.
- **Quadratic readouts** of a field F are cᵀGc, with G_ik = Σ_cells Fᵢ·F_k.
- **No sampling:** all 26 non-zero weight vectors (13 sign classes) are
  evaluated exactly. The overhead ‖c‖₁ is recorded as a label only.

Predictions were sealed in [`PREREG-VIRTUAL.hbp`](PREREG-VIRTUAL.hbp) (sha256
`dfcfb977…`, commit `358b37b`) before the code was written. The run replays the
committed experiment and asserts it against `results/` as it goes: all 228
committed series rows and the three intake totals match. Receipts:
[`results-virtual/`](results-virtual/).

![Correlation of the three lattices over time](results-virtual/virtual-figure.png)

**Figure.** Pairwise correlation, −1000…+1000‰ (gold A·S, blue A·U, grey S·U).
- **Top:** the vortex (curl) fields with the wind on.
- **Bottom:** the wind-response momentum fields (on − off).

### Verdicts

| ID | Prediction | Measured | Verdict |
|---|---|---|---|
| V1 | virtual intake = Σ cᵢ × branch totals, all 26 | holds for all 26 | **PASS** |
| V2 | Gram diagonals = the committed branch enstrophies at every checkpoint | 228 of 228 rows | **PASS** |
| V3 | every virtual quadratic readout ≥ 0 | all 26 × all checkpoints × all 6 Gram matrices | **PASS** |
| V4 | wind-response momentum correlation ≥ 900‰ for every pair at t = 576 | 803, 817, 819‰ | **FAIL** |
| V5 | wind-off vortex correlation A·S < 500‰ at t = 576 | −1‰ | **PASS** |
| V6 | the T3 minimiser is balanced (Σc = 0) | c = (−1, +1, 0) | **PASS** |

### What the virtual layer shows

**The vortex fields are different in every lattice.**

| t | 0 | 8 | 32 | 64 | 576 |
|---|---|---|---|---|---|
| vortex correlation A·S (wind off) | 370‰ | 40‰ | 14‰ | 8‰ | −1‰ |
| vortex correlation A·U (wind off) | 590‰ | 99‰ | 34‰ | 14‰ | 0‰ |

- At t = 0 the three lattices share part of their vortex field: the common
  Taylor–Green imprint.
- Within 8 ticks the shared part is gone, and by t = 576 the three vortex fields
  are uncorrelated.
- So although Asolaria and its shuffled twin take in the same wind momentum to
  within 0.66% (§2.2), their vortex patterns have nothing in common. The
  arrangement of the GGUF trits sets *where* the vortices are, not how many.

**The wind's push is shared.** The correlation of the wind-response momentum
fields rises from 0 to 803–819‰. It does not reach the 900‰ that V4 predicted.

**The tasks** (the paper's three, exhaustive over the 26 weights):
- **T1, cancellation:** no balanced weight cancels the wind-on vortex field; the
  best, (0, +1, −1), leaves 999‰ of the uncorrelated value.
- **T2, coherence:** for c = (1, 1, 1), the wind-response momentum adds coherently
  (2,613‰ of the uncorrelated sum, out of a possible 3,000‰). The wind-response
  vortex fields add incoherently (999‰).
- **T3, virtual ground state:** c = (−1, +1, 0), shuffled minus Asolaria, has a
  wind-on flow energy of 7,162,412. That is 6.9× below the smallest single
  lattice (49,318,776): the two flows cancel 85%.

## 4. Reproduce

```sh
cargo clippy --release --all-targets -- -D warnings
cargo test --release
cargo run --release -- all data out 576
cd out && sha256sum -c ../results/SHA256SUMS && cd ..
cargo run --release -- virtual data results vout 576
cd vout && sha256sum -c ../results-virtual/SHA256SUMS
```

Each run takes about 8½ minutes on one core. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
installs Rust 1.81.0 and then:
- fails on any `f32`/`f64`, JSON, JS or Node file;
- runs clippy and the tests;
- checks the 27 GGUF inputs against the manifest;
- reruns the full experiment and checks every output against the committed
  `results/SHA256SUMS`;
- reruns the virtual layer and checks it against `results-virtual/SHA256SUMS`.

| Path | Contents |
|---|---|
| `PREREG.hbp` | the sealed predictions |
| `data/` | the 27 GGUFs and `MANIFEST.hbp` |
| `src/gguf.rs` | GGUF v3 reader, integer magnitudes |
| `src/field.rs` | GGUF → 64³ trits |
| `src/lattice.rs` | D3Q27 gates, streaming, inverse |
| `src/physics.rs` | Taylor–Green, solar wind, enstrophy, front, hashes |
| `src/sha256.rs`, `src/hbp.rs`, `src/png.rs` | SHA-256, HBP/HBI receipts, integer PNG writer |
| `src/main.rs` | the experiment |
| `tests/integer_network.rs`, `tests/virtual_layer.rs` | SHA-256 vectors, the gate-table hash, GGUF reading, conservation |
| `src/virtual_layer.rs`, `src/virtual_run.rs` | the balanced-trit weights, Gram matrices, exact readouts; the `virtual` command |
| `results-virtual/` | `VIRTUAL.hbp` (verdicts, tasks, the 13 sign-class rows), `VIRTUAL-SERIES.hbp` (all six Gram matrices every 8 ticks), `virtual-figure.png`, `INDEX.hbi`, `SHA256SUMS` |
| `results/` | `RESULT.hbp`, `SERIES.hbp` (every 8 ticks, all 27 outer modes), `CHANNELS.hbp`, `INDEX.hbi`, `SHA256SUMS`, `lattice.trits`, `figure.png` |

Operator: Jesse Daniel Brown.
