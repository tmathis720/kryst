#[derive(Clone, Copy, Debug)]
pub(crate) enum ComparisonConfidence {
    Exact,
    Approximate,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IterationRange {
    pub(crate) min: usize,
    pub(crate) max: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixExpectation {
    pub(crate) matrix: &'static str,
    pub(crate) solver_families: &'static [&'static str],
    pub(crate) preconditioner_families: &'static [&'static str],
    pub(crate) iteration_range: IterationRange,
    pub(crate) time_note: Option<&'static str>,
    pub(crate) confidence: ComparisonConfidence,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BenchmarkDelta {
    pub(crate) method_family_match: bool,
    pub(crate) iteration_delta_low: isize,
    pub(crate) iteration_delta_high: isize,
    pub(crate) tolerance_pass: bool,
}

const CATALOG: &[MatrixExpectation] = &[
    MatrixExpectation {
        matrix: "fidap005",
        solver_families: &["cg"],
        preconditioner_families: &["amg"],
        iteration_range: IterationRange { min: 70, max: 140 },
        time_note: Some("prefer AMG setup reuse when running repeated solves"),
        confidence: ComparisonConfidence::Approximate,
    },
    MatrixExpectation {
        matrix: "e05r0100",
        solver_families: &["gmres"],
        preconditioner_families: &["none"],
        iteration_range: IterationRange { min: 180, max: 320 },
        time_note: Some("no-preconditioner baseline is usually fastest on this case"),
        confidence: ComparisonConfidence::Approximate,
    },
    MatrixExpectation {
        matrix: "fidap001",
        solver_families: &["gmres"],
        preconditioner_families: &["ilu", "ilut"],
        iteration_range: IterationRange { min: 350, max: 700 },
        time_note: None,
        confidence: ComparisonConfidence::Approximate,
    },
    MatrixExpectation {
        matrix: "sherman3",
        solver_families: &["gmres", "fgmres"],
        preconditioner_families: &["ilu", "ilut"],
        iteration_range: IterationRange { min: 20, max: 80 },
        time_note: Some("ILU-family preconditioners should dominate wall time tradeoffs"),
        confidence: ComparisonConfidence::Approximate,
    },
    MatrixExpectation {
        matrix: "add20",
        solver_families: &["cg", "bicgstab", "gmres"],
        preconditioner_families: &["ilu", "ilut"],
        iteration_range: IterationRange { min: 2, max: 10 },
        time_note: None,
        confidence: ComparisonConfidence::Exact,
    },
    MatrixExpectation {
        matrix: "memplus",
        solver_families: &["bicgstab", "gmres"],
        preconditioner_families: &["ilu", "ilut"],
        iteration_range: IterationRange { min: 3, max: 12 },
        time_note: Some("without ILU-family preconditioning this matrix may stall badly"),
        confidence: ComparisonConfidence::Approximate,
    },
];

pub(crate) fn expectation_for(matrix_name: &str) -> Option<&'static MatrixExpectation> {
    CATALOG.iter().find(|entry| entry.matrix == matrix_name)
}

pub(crate) fn compare_best_iterative(
    matrix_name: &str,
    solver: &str,
    preconditioner: &str,
    iterations: usize,
    accepted: bool,
) -> Option<BenchmarkDelta> {
    let expectation = expectation_for(matrix_name)?;
    let method_family_match = family_match(solver, expectation.solver_families)
        && family_match(preconditioner, expectation.preconditioner_families);

    let iteration_delta_low = iterations as isize - expectation.iteration_range.min as isize;
    let iteration_delta_high = iterations as isize - expectation.iteration_range.max as isize;
    let within_range = iterations >= expectation.iteration_range.min
        && iterations <= expectation.iteration_range.max;
    let tolerance_pass = match expectation.confidence {
        ComparisonConfidence::Exact => accepted && method_family_match && within_range,
        ComparisonConfidence::Approximate => {
            accepted
                && method_family_match
                && (within_range || approx_margin_hit(iterations, expectation.iteration_range))
        }
    };

    Some(BenchmarkDelta {
        method_family_match,
        iteration_delta_low,
        iteration_delta_high,
        tolerance_pass,
    })
}

fn normalize_family(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn family_match(actual: &str, expected_families: &[&str]) -> bool {
    let actual_norm = normalize_family(actual);
    expected_families.iter().any(|family| {
        let family_norm = normalize_family(family);
        actual_norm.contains(&family_norm) || family_norm.contains(&actual_norm)
    })
}

fn approx_margin_hit(iterations: usize, range: IterationRange) -> bool {
    let width = range.max.saturating_sub(range.min).max(1);
    let margin = ((width as f64) * 0.25).ceil() as usize;
    let low = range.min.saturating_sub(margin);
    let high = range.max.saturating_add(margin);
    iterations >= low && iterations <= high
}
