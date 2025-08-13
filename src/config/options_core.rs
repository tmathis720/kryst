//! Minimal, table-driven options engine for PETSc-style flags.
//! Single pass over argv, with fuzzy suggestions, boolean toggles,
//! options-file expansion, and help generation.

use std::{collections::HashMap, fmt::Display, fs, path::Path, str::FromStr};

use crate::error::KError;

#[derive(Copy, Clone, Debug)]
pub enum Arity {
    Zero, // presence toggles true (also supports explicit true/false/1/0)
    One,
    Two,
}

#[derive(Copy, Clone, Debug)]
pub enum ValueKind {
    Bool,
    Int,
    UInt,
    Float,
    Str,
    Pair(&'static str, &'static str), // only used for Arity::Two
}

#[derive(Copy, Clone, Debug)]
pub struct Spec {
    pub flag: &'static str, // e.g., "-ksp_rtol"
    pub key:  &'static str, // mapping key used by Sink ("ksp_rtol")
    pub arity: Arity,
    pub kind:  ValueKind,   // used for help + error messaging
    /// Optional doc blurb for generated help.
    pub doc:   &'static str,
}

#[derive(Debug)]
pub struct Registry {
    by_flag: HashMap<&'static str, Spec>,
    flags:   Vec<&'static str>,
}

impl Registry {
    pub fn new(specs: &'static [Spec]) -> Self {
        let mut by_flag = HashMap::with_capacity(specs.len());
        for s in specs {
            by_flag.insert(s.flag, *s);
        }
        Self {
            by_flag,
            flags: specs.iter().map(|s| s.flag).collect(),
        }
    }

    /// Parse argv into Sink, optionally filtering by prefix (e.g., "-ksp_", "-pc_").
    pub fn parse_into(
        &self,
        args: &[&str],
        sink: &mut dyn Sink,
        prefix_filter: Option<&str>,
    ) -> Result<(), KError> {
        let mut i = 0usize;
        while i < args.len() {
            let tok = args[i];
            let looks_like_flag = tok.starts_with('-');
            if !looks_like_flag
                || prefix_filter.map_or(false, |p| !tok.starts_with(p))
            {
                i += 1;
                continue;
            }
            let Some(spec) = self.by_flag.get(tok) else {
                // Unknown flag that looks like ours: suggest close match
                let guess = nearest(tok, &self.flags);
                let mut msg = format!("Unrecognized option: {tok}");
                if let Some(g) = guess { msg.push_str(&format!(" (did you mean {g}?)")); }
                return Err(KError::SolveError(msg));
            };

            match spec.arity {
                Arity::Zero => {
                    // presence implies true; allow optional explicit bool token
                    let val = match args.get(i + 1).map(|s| s.to_lowercase()) {
                        Some(ref s) if is_bool_literal(s) => { i += 1; parse_bool(s)? }
                        _ => true,
                    };
                    sink.set_bool(spec.key, val)?;
                    i += 1;
                }
                Arity::One => {
                    let Some(v) = args.get(i + 1) else {
                        return Err(KError::SolveError(format!("Missing value for {}", spec.flag)));
                    };
                    sink.set_val(spec, v)?;
                    i += 2;
                }
                Arity::Two => {
                    let (a, b) = (args.get(i + 1), args.get(i + 2));
                    if a.is_none() || b.is_none() {
                        return Err(KError::SolveError(format!("Missing values for {} (needs two)", spec.flag)));
                    }
                    sink.set_pair(spec, a.unwrap(), b.unwrap())?;
                    i += 3;
                }
            }
        }
        Ok(())
    }

    pub fn help_for_prefix(&self, prefix: &str) -> String {
        let mut items: Vec<_> = self.by_flag.values()
            .filter(|s| s.flag.starts_with(prefix))
            .collect();
        items.sort_by_key(|s| s.flag);
        let mut out = String::new();
        for s in items {
            let ar = match s.arity { Arity::Zero => "", Arity::One => " <val>", Arity::Two => " <a> <b>" };
            out.push_str(&format!("  {:<34} {:<8} {}\n", format!("{}{}", s.flag, ar), kind_str(s.kind), s.doc));
        }
        out
    }
}

fn kind_str(k: ValueKind) -> &'static str {
    match k {
        ValueKind::Bool => "bool",
        ValueKind::Int => "int",
        ValueKind::UInt => "uint",
        ValueKind::Float => "float",
        ValueKind::Str => "str",
        ValueKind::Pair(a, b) => {
            // e.g., "uint,uint"
            if a.is_empty() && b.is_empty() { "pair" } else { "pair" }
        }
    }
}

fn is_bool_literal(s: &str) -> bool {
    matches!(s, "true"|"false"|"1"|"0"|"yes"|"no"|"on"|"off")
}
fn parse_bool(s: &str) -> Result<bool, KError> {
    Ok(match s {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        _ => return Err(KError::SolveError(format!("Invalid boolean literal: {s}"))),
    })
}

/// A sink is the typed receiver for parsed options (KspOptions, PcOptions).
pub trait Sink {
    fn set_bool(&mut self, key: &str, v: bool) -> Result<(), KError>;
    fn set_val(&mut self, spec: &Spec, v: &str) -> Result<(), KError>;
    fn set_pair(&mut self, spec: &Spec, a: &str, b: &str) -> Result<(), KError>;
}

/// Expand `-options_file <path>` occurrences (PETSc-style).
/// Lines starting with `#` are comments. Splits by ASCII whitespace.
/// Returns a flattened argv vector (no recursion limit, but file includes are not re-expanded inside files).
pub fn expand_options_files(mut args: Vec<String>) -> Result<Vec<String>, KError> {
    let mut out = Vec::<String>::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "-options_file" {
            let path = args.get(i + 1)
                .ok_or_else(|| KError::SolveError("Missing value for -options_file".into()))?;
            let file_args = read_options_file(Path::new(path))?;
            out.extend(file_args);
            i += 2;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    Ok(out)
}

fn read_options_file(path: &Path) -> Result<Vec<String>, KError> {
    let text = fs::read_to_string(path)
        .map_err(|e| KError::SolveError(format!("Failed to read options file {:?}: {e}", path)))?;
    let mut toks = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        toks.extend(line.split_whitespace().map(|s| s.to_string()));
    }
    Ok(toks)
}

// bounded Levenshtein for suggestions
fn nearest<'a>(needle: &str, hay: &[&'a str]) -> Option<&'a str> {
    fn dist(a: &str, b: &str) -> usize {
        let (m, n) = (a.len(), b.len());
        let mut prev = (0..=n).collect::<Vec<_>>();
        for (i, ca) in a.chars().enumerate() {
            let mut curr = vec![i + 1];
            for (j, cb) in b.chars().enumerate() {
                let ins = curr[j] + 1;
                let del = prev[j + 1] + 1;
                let sub = prev[j] + usize::from(ca != cb);
                curr.push(ins.min(del).min(sub));
            }
            prev = curr;
        }
        *prev.last().unwrap()
    }
    hay.iter().copied().min_by_key(|&cand| dist(needle, cand)).and_then(|cand| {
        if dist(needle, cand) <= 3 { Some(cand) } else { None }
    })
}

// Generic parse helper
pub fn parse_as<T: FromStr>(s: &str, spec: &Spec) -> Result<T, KError>
where <T as FromStr>::Err: Display
{
    s.parse::<T>().map_err(|e| {
        KError::SolveError(format!("Invalid value for {} ({}): {} ({e})", spec.flag, kind_str(spec.kind), s))
    })
}

/// Check if help is requested in the arguments.
pub fn is_help_requested(args: &[&str]) -> bool {
    args.iter().any(|&arg| arg == "-help" || arg == "--help" || arg == "-h")
}
