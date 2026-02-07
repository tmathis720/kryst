//! Minimal, table-driven options engine for PETSc-style flags.
//! Single pass over argv, with fuzzy suggestions, boolean toggles,
//! options-file expansion, and help generation.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::error::KError;

#[derive(Copy, Clone, Debug)]
pub enum Arity {
    Zero,         // presence toggles true (also supports explicit true/false/1/0)
    OptionalBool, // presence toggles true but behaves like Zero; flagged explicitly in the registry
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
    pub key: &'static str,  // mapping key used by Sink ("ksp_rtol")
    pub arity: Arity,
    pub kind: ValueKind, // used for help + error messaging
    /// Optional doc blurb for generated help.
    pub doc: &'static str,
}

#[derive(Debug)]
pub struct Registry {
    by_flag: HashMap<&'static str, Spec>,
    flags: Vec<&'static str>,
}

impl Registry {
    pub fn new(specs: &[Spec]) -> Self {
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
            if !looks_like_flag || prefix_filter.is_some_and(|p| !tok.starts_with(p)) {
                i += 1;
                continue;
            }
            let spec = if let Some(spec) = self.by_flag.get(tok) {
                *spec
            } else if let Some(pref) = prefix_filter {
                if tok.starts_with(pref) {
                    let suffix = &tok[pref.len()..];
                    let canonical = format!("-{suffix}");
                    if let Some(spec) = self.by_flag.get(canonical.as_str()) {
                        *spec
                    } else if is_hierarchical_flag(tok, pref, &self.flags) {
                        i += 1;
                        continue;
                    } else {
                        let guess = nearest(canonical.as_str(), &self.flags);
                        let mut msg = format!("Unrecognized option: {tok}");
                        if let Some(g) = guess {
                            msg.push_str(&format!(" (did you mean {} with prefix?)", g));
                        }
                        return Err(KError::SolveError(msg));
                    }
                } else {
                    let guess = nearest(tok, &self.flags);
                    let mut msg = format!("Unrecognized option: {tok}");
                    if let Some(g) = guess {
                        msg.push_str(&format!(" (did you mean {g}?)"));
                    }
                    return Err(KError::SolveError(msg));
                }
            } else {
                let guess = nearest(tok, &self.flags);
                let mut msg = format!("Unrecognized option: {tok}");
                if let Some(g) = guess {
                    msg.push_str(&format!(" (did you mean {g}?)"));
                }
                return Err(KError::SolveError(msg));
            };

            match spec.arity {
                Arity::Zero | Arity::OptionalBool => {
                    // presence implies true; allow optional explicit bool token
                    let val = match args.get(i + 1).map(|s| s.to_lowercase()) {
                        Some(ref s) if is_bool_literal(s) => {
                            i += 1;
                            parse_bool(s)?
                        }
                        _ => true,
                    };
                    sink.set_bool(spec.key, val)?;
                    i += 1;
                }
                Arity::One => {
                    let Some(v) = args.get(i + 1) else {
                        return Err(KError::SolveError(format!(
                            "Missing value for {}",
                            spec.flag
                        )));
                    };
                    sink.set_val(&spec, v)?;
                    i += 2;
                }
                Arity::Two => {
                    let (a, b) = (args.get(i + 1), args.get(i + 2));
                    if a.is_none() || b.is_none() {
                        return Err(KError::SolveError(format!(
                            "Missing values for {} (needs two)",
                            spec.flag
                        )));
                    }
                    sink.set_pair(&spec, a.unwrap(), b.unwrap())?;
                    i += 3;
                }
            }
        }
        Ok(())
    }

    pub fn help_for_prefix(&self, prefix: &str) -> String {
        let mut items: Vec<_> = self
            .by_flag
            .values()
            .filter(|s| s.flag.starts_with(prefix))
            .collect();
        items.sort_by_key(|s| s.flag);
        let mut out = String::new();
        for s in items {
            let ar = match s.arity {
                Arity::Zero | Arity::OptionalBool => "",
                Arity::One => " <val>",
                Arity::Two => " <a> <b>",
            };
            out.push_str(&format!(
                "  {:<34} {:<8} {}\n",
                format!("{}{}", s.flag, ar),
                kind_str(s.kind),
                s.doc
            ));
        }
        out
    }

    pub fn spec_for_flag(&self, flag: &str) -> Option<Spec> {
        self.by_flag.get(flag).copied()
    }
}

fn is_hierarchical_flag(tok: &str, prefix: &str, flags: &[&'static str]) -> bool {
    let Some(suffix) = tok.strip_prefix(prefix) else {
        return false;
    };
    if !suffix.contains('_') {
        return false;
    }
    flags.iter().any(|flag| {
        let flag_suffix = flag.strip_prefix('-').unwrap_or(flag);
        suffix != flag_suffix && suffix.ends_with(&format!("_{flag_suffix}"))
    })
}

fn kind_str(k: ValueKind) -> Cow<'static, str> {
    match k {
        ValueKind::Bool => Cow::Borrowed("bool"),
        ValueKind::Int => Cow::Borrowed("int"),
        ValueKind::UInt => Cow::Borrowed("uint"),
        ValueKind::Float => Cow::Borrowed("float"),
        ValueKind::Str => Cow::Borrowed("str"),
        ValueKind::Pair(a, b) => Cow::Owned(format!("{a},{b}")),
    }
}

fn is_bool_literal(s: &str) -> bool {
    matches!(
        s,
        "true" | "false" | "1" | "0" | "yes" | "no" | "on" | "off"
    )
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
/// Returns a flattened argv vector after recursively expanding nested includes.
/// Relative include paths are resolved against the including file’s directory.
/// Detects include cycles and errors out with a friendly message.
pub fn expand_options_files(args: Vec<String>) -> Result<Vec<String>, KError> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut stack = Vec::<PathBuf>::new();
    expand_tokens_recursive(&args, &cwd, &mut stack)
}

// Recursively expand -options_file tokens found in `tokens`,
// using `base_dir` for resolving relative include paths.
fn expand_tokens_recursive(
    tokens: &[String],
    base_dir: &Path,
    stack: &mut Vec<PathBuf>,
) -> Result<Vec<String>, KError> {
    let mut out = Vec::<String>::new();
    let mut i = 0usize;

    while i < tokens.len() {
        if tokens[i] == "-options_file" {
            let path_str = tokens
                .get(i + 1)
                .ok_or_else(|| KError::SolveError("Missing value for -options_file".into()))?;

            let mut path = PathBuf::from(path_str);
            if path.is_relative() {
                path = base_dir.join(&path);
            }
            // Canonicalize for reliable cycle detection, but fall back to joined path
            let canon = path.canonicalize().unwrap_or_else(|_| path.clone());

            // Cycle detection: A -> B -> A
            if let Some(pos) = stack.iter().position(|p| *p == canon) {
                // Build a readable cycle chain
                let mut chain: Vec<String> = stack[pos..]
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect();
                chain.push(canon.display().to_string());
                return Err(KError::SolveError(format!(
                    "Cyclic -options_file include detected: {}",
                    chain.join(" -> ")
                )));
            }

            stack.push(canon.clone());
            let file_tokens = read_options_file(&canon)?; // tokenize the file’s content
            let next_base = canon.parent().unwrap_or(base_dir);
            let expanded = expand_tokens_recursive(&file_tokens, next_base, stack)?;
            stack.pop();

            out.extend(expanded);
            i += 2;
        } else {
            out.push(tokens[i].clone());
            i += 1;
        }
    }

    Ok(out)
}

fn read_options_file(path: &Path) -> Result<Vec<String>, KError> {
    let text = fs::read_to_string(path)
        .map_err(|e| KError::SolveError(format!("Failed to read options file {path:?}: {e}")))?;
    let mut toks = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        toks.extend(line.split_whitespace().map(|s| s.to_string()));
    }
    Ok(toks)
}

// bounded Levenshtein for suggestions
fn nearest<'a>(needle: &str, hay: &[&'a str]) -> Option<&'a str> {
    fn dist(a: &str, b: &str) -> usize {
        let (_m, n) = (a.len(), b.len());
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
    hay.iter()
        .copied()
        .min_by_key(|&cand| dist(needle, cand))
        .and_then(|cand| {
            if dist(needle, cand) <= 3 {
                Some(cand)
            } else {
                None
            }
        })
}

// Generic parse helper
pub fn parse_as<T: FromStr>(s: &str, spec: &Spec) -> Result<T, KError>
where
    <T as FromStr>::Err: Display,
{
    s.parse::<T>().map_err(|e| {
        KError::SolveError(format!(
            "Invalid value for {} ({}): {} ({e})",
            spec.flag,
            kind_str(spec.kind),
            s
        ))
    })
}

/// Check if help is requested in the arguments.
pub fn is_help_requested(args: &[&str]) -> bool {
    args.iter()
        .any(|&arg| arg == "-help" || arg == "--help" || arg == "-h")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_str_simple_kinds_are_borrowed() {
        match kind_str(ValueKind::UInt) {
            Cow::Borrowed(s) => assert_eq!(s, "uint"),
            Cow::Owned(_) => panic!("expected borrowed for simple kind"),
        }
    }

    #[test]
    fn kind_str_pair_joins_inner_strings() {
        let k = ValueKind::Pair("uint", "uint");
        assert_eq!(kind_str(k), "uint,uint");
    }

    #[test]
    fn help_includes_precise_pair_kind() {
        let specs = [Spec {
            flag: "-example_pair",
            key: "example_pair",
            arity: Arity::Two,
            kind: ValueKind::Pair("width", "height"),
            doc: "example pair flag",
        }];
        let reg = Registry::new(&specs);
        let help = reg.help_for_prefix("-example_");
        assert!(help.contains("width,height"), "help was: {help}");
        // Ensure generic 'pair' is not shown
        assert!(!help.contains(" pair\n"));
    }
}
