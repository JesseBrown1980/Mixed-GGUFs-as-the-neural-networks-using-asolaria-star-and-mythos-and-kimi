//! HBP receipt lines: KEY|k=v|...|json=0, and HBI index rows.

pub struct Hbp {
    lines: Vec<String>,
}

impl Default for Hbp {
    fn default() -> Self {
        Self::new()
    }
}

impl Hbp {
    pub fn new() -> Self {
        Hbp { lines: Vec::new() }
    }
    pub fn row(&mut self, key: &str, fields: &[(&str, String)]) -> &str {
        let mut s = String::from(key);
        for (k, v) in fields {
            s.push('|');
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s.push_str("|json=0");
        self.lines.push(s);
        self.lines.last().map(|x| x.as_str()).unwrap_or("")
    }
    pub fn text(&self) -> String {
        let mut s = self.lines.join("\n");
        s.push('\n');
        s
    }
    pub fn len(&self) -> usize {
        self.lines.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// HBI index over a set of files: one INDEXROW per file with its byte count and sha256.
pub fn index(name: &str, files: &[(String, usize, String)]) -> String {
    let mut s = format!("{}|files={}|json=0\n", name, files.len());
    for (i, (f, n, h)) in files.iter().enumerate() {
        s.push_str(&format!(
            "INDEXROW|row={}|file={}|bytes={}|sha256={}|json=0\n",
            i + 1,
            f,
            n,
            h
        ));
    }
    s
}
