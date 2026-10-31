//! GGUF v3 reader. Tensor elements come back as non-negative integer magnitudes:
//! F32 -> raw IEEE-754 bits with the sign bit cleared (orders exactly like |v|);
//! I8 -> the unsigned byte. No floating-point value is ever formed.

use std::fmt;

#[derive(Debug)]
pub struct GgufError(pub String);
impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub mags: Vec<u32>,
}

pub struct Gguf {
    pub version: u32,
    pub n_kv: u64,
    pub tensors: Vec<Tensor>,
    pub bytes_read_to: usize,
}

struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .o
            .checked_add(n)
            .filter(|e| *e <= self.b.len())
            .ok_or_else(|| GgufError(format!("truncated at {}", self.o)))?;
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    fn len(&mut self) -> Result<usize, GgufError> {
        usize::try_from(self.u64()?).map_err(|_| GgufError("length".into()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn skip_value(&mut self, t: u32) -> Result<(), GgufError> {
        match t {
            0 | 1 | 7 => {
                self.take(1)?;
            }
            2 | 3 => {
                self.take(2)?;
            }
            4..=6 => {
                self.take(4)?;
            }
            10..=12 => {
                self.take(8)?;
            }
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
            }
            _ => return Err(GgufError(format!("kv type {t}"))),
        }
        Ok(())
    }
}

pub fn parse(b: &[u8]) -> Result<Gguf, GgufError> {
    let mut c = Cur { b, o: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(GgufError("magic".into()));
    }
    let version = c.u32()?;
    let n_t = c.u64()?;
    let n_kv = c.u64()?;
    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = c.string()?;
        let t = c.u32()?;
        if key == "general.alignment" && t == 4 {
            alignment = u64::from(c.u32()?);
        } else {
            c.skip_value(t)?;
        }
    }
    let mut infos = Vec::new();
    for _ in 0..n_t {
        let name = c.string()?;
        let nd = c.u32()?;
        let mut dims = Vec::new();
        for _ in 0..nd {
            dims.push(c.u64()?);
        }
        let t = c.u32()?;
        let off = c.u64()?;
        infos.push((name, dims, t, off));
    }
    let a = usize::try_from(alignment).map_err(|_| GgufError("alignment".into()))?;
    let base = c.o.div_ceil(a) * a;
    let mut tensors = Vec::new();
    let mut end_max = base;
    for (name, dims, t, off) in infos {
        let count: u64 = dims.iter().product();
        let count = usize::try_from(count).map_err(|_| GgufError("count".into()))?;
        let width = match t {
            0 => 4,
            24 => 1,
            _ => return Err(GgufError(format!("{name}: ggml type {t}"))),
        };
        let start = base + usize::try_from(off).map_err(|_| GgufError("off".into()))?;
        let end = start + count * width;
        if end > b.len() {
            return Err(GgufError(format!("{name}: past end")));
        }
        let raw = &b[start..end];
        let mags: Vec<u32> = if width == 4 {
            raw.chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) & 0x7FFF_FFFF)
                .collect()
        } else {
            raw.iter().map(|&x| u32::from(x)).collect()
        };
        end_max = end_max.max(end);
        tensors.push(Tensor {
            name,
            dims,
            ggml_type: t,
            mags,
        });
    }
    Ok(Gguf {
        version,
        n_kv,
        tensors,
        bytes_read_to: end_max,
    })
}
