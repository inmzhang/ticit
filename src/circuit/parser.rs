//! Parser for the line-oriented `.ticit` circuit format.
//!
//! # Shape of the parse
//!
//! Two passes over a line-oriented source:
//!
//! 1. `parse_nodes` turns lines into a tree of instructions and `REPEAT`
//!    blocks. This is the only place block structure exists.
//! 2. `max_qubit_index` walks that tree to size the circuit's Pauli strings,
//!    then `append_nodes` walks it again, expanding every `REPEAT` by literal
//!    repetition and lowering each instruction into IR.
//!
//! Sizing has to happen first because `MPP X0*Z1` builds a [`PauliString`] whose
//! width is the whole circuit's qubit count.
//!
//! Every parse error carries its 1-based source line. Numeric targets use
//! nonnegative decimal indices, and instruction tags are accepted but ignored.

use std::f64::consts::PI;
use std::path::Path;

use crate::circuit::ir::{
    Circuit, CircuitDetector, CircuitFeedbackTarget, CircuitInstruction, CircuitInstructionKind,
    CircuitMeasurementTarget, CircuitObservableInclude, CircuitPauliProduct,
};
use crate::errors::{Result, TicitError};
use crate::pauli::{PauliString, pauli_identity, pauli_x, pauli_y, pauli_z};

// ==============================================================================
// Public entry points
// ==============================================================================

/// Parses a Ticit circuit from a sequence of lines.
pub fn parse_ticit_circuit_lines<S: AsRef<str>>(lines: &[S]) -> Result<Circuit> {
    let mut pos = 0;
    let nodes = parse_nodes(lines, &mut pos, false)?;
    let mut builder = TicitCircuitBuilder::default();
    builder.circuit.nqubits = max_qubit_index(&nodes).map_or(0, |max| max + 1);
    append_nodes(&mut builder, &nodes)?;
    Ok(builder.circuit)
}

/// Parses a Ticit circuit from source text.
pub fn parse_ticit_circuit_text(text: &str) -> Result<Circuit> {
    let lines: Vec<&str> = text.lines().collect();
    parse_ticit_circuit_lines(&lines)
}

/// Parses a Ticit circuit from a file.
pub fn parse_ticit_circuit_file(path: impl AsRef<Path>) -> Result<Circuit> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| TicitError::io(path, source))?;
    parse_ticit_circuit_text(&text)
}

// ==============================================================================
// Errors
// ==============================================================================

/// Builds a parse error tagged with its source line.
fn err_at(line: usize, message: impl std::fmt::Display) -> TicitError {
    TicitError::parse(line, message)
}

// ==============================================================================
// Lexical helpers
// ==============================================================================

fn all_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// Parses a nonnegative decimal index, rejecting signs and trailing garbage.
fn parse_index(value: &str) -> Option<usize> {
    all_digits(value).then(|| value.parse().ok()).flatten()
}

/// Drops a `#` comment and surrounding whitespace. `#` inside `[...]` is kept,
/// so a tag may contain one; the flag is not nesting-aware, matching the C++.
fn strip_ticit_comment(line: &str) -> &str {
    let mut in_brackets = false;
    for (index, byte) in line.bytes().enumerate() {
        match byte {
            b'[' => in_brackets = true,
            b']' => in_brackets = false,
            b'#' if !in_brackets => return line[..index].trim(),
            _ => {}
        }
    }
    line.trim()
}

fn find_matching_paren(body: &str, open: usize, line: usize) -> Result<usize> {
    let mut depth = 0i32;
    for (index, byte) in body.bytes().enumerate().skip(open) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(index);
                }
            }
            _ => {}
        }
    }
    Err(err_at(line, "unterminated Ticit argument list"))
}

/// Splits an argument list on top-level commas. The trailing element is emitted
/// by a virtual comma past the end, so `"1,"` yields `["1", ""]` and the empty
/// piece fails later in numeric parsing.
fn split_arguments(args: &str, line: usize) -> Result<Vec<&str>> {
    let bytes = args.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    for index in 0..=bytes.len() {
        let byte = if index < bytes.len() {
            bytes[index]
        } else {
            b','
        };
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    return Err(err_at(line, "invalid numeric argument nesting"));
                }
            }
            b',' if depth == 0 => {
                out.push(args[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(err_at(line, "invalid numeric argument nesting"));
    }
    Ok(out)
}

// ==============================================================================
// Numeric expression parser
// ==============================================================================

/// Longest prefix of `bytes` from `start` that is a C-locale decimal float.
///
/// Returns the end offset, or `None` when no digits are present. An `e` not
/// followed by digits is left unconsumed, so `"1e"` scans as `"1"` — the same
/// longest-valid-prefix rule `strtod` uses.
fn scan_number(bytes: &[u8], start: usize) -> Option<usize> {
    let mut pos = start;
    if matches!(bytes.get(pos), Some(b'+' | b'-')) {
        pos += 1;
    }
    let integer_start = pos;
    while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
        pos += 1;
    }
    let mut has_digits = pos > integer_start;
    if bytes.get(pos) == Some(&b'.') {
        pos += 1;
        let fraction_start = pos;
        while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        has_digits |= pos > fraction_start;
    }
    if !has_digits {
        return None;
    }
    if matches!(bytes.get(pos), Some(b'e' | b'E')) {
        let mut exponent = pos + 1;
        if matches!(bytes.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        let exponent_digits = exponent;
        while bytes.get(exponent).is_some_and(u8::is_ascii_digit) {
            exponent += 1;
        }
        if exponent > exponent_digits {
            pos = exponent;
        }
    }
    Some(pos)
}

/// Recursive-descent evaluator for paren arguments: `+ - * /`, unary sign,
/// grouping, and the single named constant `PI`.
struct ArgumentExpressionParser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> ArgumentExpressionParser<'a> {
    fn new(input: &'a str, line: usize) -> Self {
        Self {
            input,
            pos: 0,
            line,
        }
    }

    fn bytes(&self) -> &'a [u8] {
        self.input.as_bytes()
    }

    fn parse(mut self) -> Result<f64> {
        let value = self.parse_expression()?;
        self.skip_whitespace();
        if self.pos != self.input.len() {
            return Err(err_at(self.line, "invalid numeric expression"));
        }
        Ok(value)
    }

    fn skip_whitespace(&mut self) {
        while self
            .bytes()
            .get(self.pos)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.pos += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        self.skip_whitespace();
        if self.bytes().get(self.pos) == Some(&expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_expression(&mut self) -> Result<f64> {
        let mut value = self.parse_term()?;
        loop {
            if self.consume(b'+') {
                value += self.parse_term()?;
            } else if self.consume(b'-') {
                value -= self.parse_term()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_term(&mut self) -> Result<f64> {
        let mut value = self.parse_factor()?;
        loop {
            if self.consume(b'*') {
                value *= self.parse_factor()?;
            } else if self.consume(b'/') {
                value /= self.parse_factor()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_factor(&mut self) -> Result<f64> {
        self.skip_whitespace();
        if self.consume(b'+') {
            return self.parse_factor();
        }
        if self.consume(b'-') {
            return Ok(-self.parse_factor()?);
        }
        if self.consume(b'(') {
            let value = self.parse_expression()?;
            if !self.consume(b')') {
                return Err(err_at(self.line, "unterminated numeric expression"));
            }
            return Ok(value);
        }
        let bytes = self.bytes();
        if bytes.get(self.pos).is_some_and(u8::is_ascii_alphabetic) {
            let start = self.pos;
            while bytes.get(self.pos).is_some_and(u8::is_ascii_alphabetic) {
                self.pos += 1;
            }
            let name = self.input[start..self.pos].to_ascii_uppercase();
            if name == "PI" {
                return Ok(PI);
            }
            return Err(err_at(
                self.line,
                format!("unknown numeric constant: {name}"),
            ));
        }
        let end = scan_number(bytes, self.pos)
            .ok_or_else(|| err_at(self.line, "invalid numeric expression"))?;
        let value = self.input[self.pos..end]
            .parse::<f64>()
            .map_err(|_| err_at(self.line, "invalid numeric expression"))?;
        self.pos = end;
        Ok(value)
    }
}

/// Evaluates one paren argument. Plain literals take a fast path; anything else
/// goes through the expression grammar.
fn parse_numeric_expression(input: &str, line: usize) -> Result<f64> {
    if let Ok(value) = input.parse::<f64>() {
        return Ok(value);
    }
    ArgumentExpressionParser::new(input, line).parse()
}

// ==============================================================================
// Instruction and block syntax
// ==============================================================================

struct TicitInstruction {
    /// Uppercased, so operation names are case-insensitive.
    op: String,
    has_parens: bool,
    parens: Vec<f64>,
    targets: Vec<String>,
    line: usize,
}

enum TicitNode {
    Instruction(TicitInstruction),
    Repeat(TicitRepeatBlock),
}

impl TicitNode {
    fn line(&self) -> usize {
        match self {
            TicitNode::Instruction(instruction) => instruction.line,
            TicitNode::Repeat(block) => block.line,
        }
    }
}

struct TicitRepeatBlock {
    count: usize,
    body: Vec<TicitNode>,
    line: usize,
}

/// Largest `REPEAT` count accepted by the format.
const MAX_REPEAT_COUNT: u128 = 1_000_000_000_000_000_000;

/// Largest repeat count representable by the flattened circuit IR.
const MAX_FLATTENED_REPEAT_COUNT: u128 = i32::MAX as u128;

fn parse_repeat_count(token: &str, line: usize) -> Result<usize> {
    let out_of_range = || err_at(line, "REPEAT count must be in [1, 10^18]");
    if !all_digits(token) {
        return Err(out_of_range());
    }
    let count: u128 = token.parse().map_err(|_| out_of_range())?;
    if count == 0 || count > MAX_REPEAT_COUNT {
        return Err(out_of_range());
    }
    if count > MAX_FLATTENED_REPEAT_COUNT {
        return Err(err_at(
            line,
            "REPEAT count is too large for this flattened circuit frontend",
        ));
    }
    Ok(count as usize)
}

fn parse_instruction(body: &str, line: usize) -> Result<TicitInstruction> {
    let bytes = body.as_bytes();

    // Operation name: letters anywhere, digits and underscores only after the
    // first character. This is what makes `U3` and `R_PAULI` single tokens.
    let mut idx = 0;
    while idx < bytes.len()
        && (bytes[idx].is_ascii_alphabetic()
            || (idx > 0 && (bytes[idx].is_ascii_digit() || bytes[idx] == b'_')))
    {
        idx += 1;
    }
    if idx == 0 {
        return Err(err_at(line, "invalid Ticit instruction"));
    }
    let op = body[..idx].to_ascii_uppercase();

    // `OP[tag]`: consumed and discarded.
    if bytes.get(idx) == Some(&b'[') {
        let close = body[idx + 1..]
            .find(']')
            .ok_or_else(|| err_at(line, "unterminated Ticit tag"))?;
        idx += 1 + close + 1;
    }

    let mut has_parens = false;
    let mut parens = Vec::new();
    if bytes.get(idx) == Some(&b'(') {
        has_parens = true;
        let close = find_matching_paren(body, idx, line)?;
        let args = body[idx + 1..close].trim();
        if !args.is_empty() {
            for piece in split_arguments(args, line)? {
                parens.push(parse_numeric_expression(piece, line)?);
            }
        }
        idx = close + 1;
    }

    let targets = body
        .get(idx..)
        .map(|rest| rest.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();

    Ok(TicitInstruction {
        op,
        has_parens,
        parens,
        targets,
        line,
    })
}

/// Builds the block tree. `pos` is shared across recursion levels so a nested
/// call resumes the outer scan where the inner `}` left off.
fn parse_nodes<S: AsRef<str>>(
    lines: &[S],
    pos: &mut usize,
    in_block: bool,
) -> Result<Vec<TicitNode>> {
    let mut nodes = Vec::new();
    while *pos < lines.len() {
        let mut body = strip_ticit_comment(lines[*pos].as_ref());
        let line = *pos + 1;
        *pos += 1;
        if body.is_empty() {
            continue;
        }
        if body == "}" {
            if !in_block {
                return Err(err_at(line, "unmatched Ticit block terminator"));
            }
            return Ok(nodes);
        }
        let block_start = body.ends_with('{');
        if block_start {
            body = body[..body.len() - 1].trim();
        }
        let instruction = parse_instruction(body, line)?;
        if block_start {
            if instruction.op != "REPEAT"
                || instruction.has_parens
                || instruction.targets.len() != 1
            {
                return Err(err_at(line, "invalid Ticit REPEAT block"));
            }
            let count = parse_repeat_count(&instruction.targets[0], line)?;
            let body = parse_nodes(lines, pos, true)?;
            nodes.push(TicitNode::Repeat(TicitRepeatBlock { count, body, line }));
        } else {
            if instruction.op == "REPEAT" {
                return Err(err_at(line, "REPEAT must start a block"));
            }
            nodes.push(TicitNode::Instruction(instruction));
        }
    }
    if in_block {
        // The block opener's line is long gone; report the end of input.
        return Err(err_at(lines.len(), "unterminated Ticit block"));
    }
    Ok(nodes)
}

// ==============================================================================
// Target grammar
// ==============================================================================

fn ticit_qubit_target(target: &str, line: usize) -> Result<(usize, bool)> {
    let (body, inverted) = match target.strip_prefix('!') {
        Some(rest) => (rest, true),
        None => (target, false),
    };
    let qubit = parse_index(body).ok_or_else(|| err_at(line, "invalid qubit target"))?;
    Ok((qubit, inverted))
}

fn ticit_qubit_targets(instruction: &TicitInstruction) -> Result<Vec<usize>> {
    instruction
        .targets
        .iter()
        .map(|target| {
            let (qubit, inverted) = ticit_qubit_target(target, instruction.line)?;
            if inverted {
                return Err(err_at(
                    instruction.line,
                    "operation does not accept inverted targets",
                ));
            }
            Ok(qubit)
        })
        .collect()
}

fn ticit_measurement_targets(instruction: &TicitInstruction) -> Result<Vec<(usize, bool)>> {
    instruction
        .targets
        .iter()
        .map(|target| ticit_qubit_target(target, instruction.line))
        .collect()
}

/// `rec[-k]`, with `k` a positive decimal.
fn record_offset(target: &str) -> Option<usize> {
    let offset = target.strip_prefix("rec[")?.strip_suffix(']')?;
    let digits = offset.strip_prefix('-')?;
    all_digits(digits).then_some(())?;
    // Out-of-range magnitudes are still record targets; they fail resolution.
    Some(digits.parse().unwrap_or(usize::MAX))
}

fn is_record_target(target: &str) -> bool {
    record_offset(target).is_some()
}

/// Resolves `rec[-k]` against the records seen so far, 1-based.
fn ticit_record_index(target: &str, nrecords: usize, line: usize) -> Result<usize> {
    let offset =
        record_offset(target).ok_or_else(|| err_at(line, "invalid measurement record target"))?;
    let index = (nrecords + 1)
        .checked_sub(offset)
        .filter(|index| (1..=nrecords).contains(index))
        .ok_or_else(|| err_at(line, "measurement record target out of range"))?;
    Ok(index)
}

fn ticit_record_indices(instruction: &TicitInstruction, nrecords: usize) -> Result<Vec<usize>> {
    instruction
        .targets
        .iter()
        .map(|target| ticit_record_index(target, nrecords, instruction.line))
        .collect()
}

fn is_sweep_target(target: &str) -> bool {
    target
        .strip_prefix("sweep[")
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(all_digits)
}

fn is_pauli_target_or_combiner(target: &str) -> bool {
    if target == "*" {
        return true;
    }
    let body = target.strip_prefix('!').unwrap_or(target);
    matches!(body.as_bytes().first(), Some(b'X' | b'Y' | b'Z')) && all_digits(&body[1..])
}

/// `OBSERVABLE_INCLUDE` collects records and silently drops Pauli targets —
/// the observable's Pauli content matters only to a state-vector consumer.
fn ticit_observable_record_indices(
    instruction: &TicitInstruction,
    nrecords: usize,
) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for target in &instruction.targets {
        if is_record_target(target) {
            out.push(ticit_record_index(target, nrecords, instruction.line)?);
        } else if !is_pauli_target_or_combiner(target) {
            return Err(err_at(
                instruction.line,
                "OBSERVABLE_INCLUDE targets must be measurement records or Pauli targets",
            ));
        }
    }
    Ok(out)
}

// ==============================================================================
// Qubit counting
// ==============================================================================

/// Largest qubit index mentioned by one target, ignoring records, sweeps and
/// combiners. Deliberately permissive: malformed targets contribute nothing and
/// are rejected later by the strict per-operation parsers.
fn target_max_qubit(target: &str) -> Option<usize> {
    if target == "*" || is_record_target(target) || is_sweep_target(target) {
        return None;
    }
    let body = target.strip_prefix('!').unwrap_or(target);
    body.split('*')
        .filter_map(|factor| {
            let factor = factor.strip_prefix('!').unwrap_or(factor);
            let digits = match factor.as_bytes().first() {
                Some(b'X' | b'Y' | b'Z') => &factor[1..],
                _ => factor,
            };
            parse_index(digits)
        })
        .max()
}

/// Annotations whose targets are not qubits: record lists, coordinate shifts,
/// and `MPAD`'s literal pad values.
fn counts_toward_qubits(op: &str) -> bool {
    !matches!(
        op,
        "DETECTOR" | "DISCARD" | "OBSERVABLE_INCLUDE" | "MPAD" | "TICK" | "SHIFT_COORDS"
    )
}

fn max_qubit_index(nodes: &[TicitNode]) -> Option<usize> {
    let mut max = None;
    for node in nodes {
        let candidate = match node {
            TicitNode::Repeat(block) => max_qubit_index(&block.body),
            TicitNode::Instruction(instruction) if counts_toward_qubits(&instruction.op) => {
                instruction
                    .targets
                    .iter()
                    .filter_map(|target| target_max_qubit(target))
                    .max()
            }
            TicitNode::Instruction(_) => None,
        };
        max = max.max(candidate);
    }
    max
}

// ==============================================================================
// Pauli product construction
// ==============================================================================

fn pauli_on_target(nqubits: usize, axis: u8, qubit: usize) -> PauliString {
    match axis {
        b'X' => pauli_x(nqubits, qubit),
        b'Y' => pauli_y(nqubits, qubit),
        _ => pauli_z(nqubits, qubit),
    }
}

/// Parses one `*`-joined Pauli product such as `X0*!Z1`. Each `!` toggles the
/// product's sign rather than a single factor's.
fn ticit_mpp_pauli(nqubits: usize, target: &str, line: usize) -> Result<(PauliString, bool)> {
    let mut out = pauli_identity(nqubits);
    let mut inverted = false;
    for factor in target.split('*') {
        let factor = match factor.strip_prefix('!') {
            Some(rest) => {
                inverted = !inverted;
                rest
            }
            None => factor,
        };
        let invalid = || err_at(line, "invalid MPP factor");
        let Some(&axis @ (b'X' | b'Y' | b'Z')) = factor.as_bytes().first() else {
            return Err(invalid());
        };
        let qubit = parse_index(&factor[1..]).ok_or_else(invalid)?;
        out = out * pauli_on_target(nqubits, axis, qubit);
    }
    Ok((out, inverted))
}

/// Regroups whitespace-separated targets around `*` combiners, so `X1 * X2`,
/// `X1* X2` and `X1*X2` all denote one product.
fn ticit_mpp_targets(instruction: &TicitInstruction) -> Result<Vec<String>> {
    let mut groups = Vec::new();
    let mut current = String::new();
    for target in &instruction.targets {
        if target == "*" {
            if current.is_empty() || current.ends_with('*') {
                return Err(err_at(instruction.line, "misplaced MPP combiner"));
            }
            current.push('*');
        } else if current.is_empty() {
            current = target.clone();
        } else if current.ends_with('*') {
            current.push_str(target);
        } else {
            groups.push(std::mem::replace(&mut current, target.clone()));
        }
    }
    if !current.is_empty() {
        if current.ends_with('*') {
            return Err(err_at(instruction.line, "dangling MPP combiner"));
        }
        groups.push(current);
    }
    Ok(groups)
}

fn circuit_mpp_targets(
    nqubits: usize,
    instruction: &TicitInstruction,
) -> Result<Vec<CircuitPauliProduct>> {
    ticit_mpp_targets(instruction)?
        .iter()
        .map(|target| {
            let (pauli, inverted) = ticit_mpp_pauli(nqubits, target, instruction.line)?;
            Ok(CircuitPauliProduct { pauli, inverted })
        })
        .collect()
}

/// `E`/`ELSE_CORRELATED_ERROR` targets form one product by juxtaposition, with
/// no combiners allowed.
fn circuit_implicit_pauli_product(
    nqubits: usize,
    instruction: &TicitInstruction,
) -> Result<CircuitPauliProduct> {
    let mut out = pauli_identity(nqubits);
    let mut inverted = false;
    for target in &instruction.targets {
        if target == "*" {
            return Err(err_at(
                instruction.line,
                "implicit Pauli products do not use combiner targets",
            ));
        }
        let (pauli, factor_inverted) = ticit_mpp_pauli(nqubits, target, instruction.line)?;
        out = out * pauli;
        inverted ^= factor_inverted;
    }
    Ok(CircuitPauliProduct {
        pauli: out,
        inverted,
    })
}

fn circuit_single_pauli_product(nqubits: usize, axis: u8, qubit: usize) -> CircuitPauliProduct {
    CircuitPauliProduct {
        pauli: pauli_on_target(nqubits, axis, qubit),
        inverted: false,
    }
}

fn circuit_pair_pauli_product(
    nqubits: usize,
    axis: u8,
    first: usize,
    second: usize,
    inverted: bool,
    line: usize,
) -> Result<CircuitPauliProduct> {
    if first == second {
        return Err(err_at(
            line,
            "two-qubit Pauli operation requires distinct qubits",
        ));
    }
    Ok(CircuitPauliProduct {
        pauli: pauli_on_target(nqubits, axis, first) * pauli_on_target(nqubits, axis, second),
        inverted,
    })
}

/// `MXX`/`MYY`/`MZZ`: consecutive target pairs, product sign = XOR of the pair's
/// `!` flags.
fn circuit_pair_measurement_products(
    nqubits: usize,
    instruction: &TicitInstruction,
    axis: u8,
) -> Result<Vec<CircuitPauliProduct>> {
    let targets = ticit_measurement_targets(instruction)?;
    if !targets.len().is_multiple_of(2) {
        return Err(err_at(
            instruction.line,
            "pair measurement requires paired targets",
        ));
    }
    targets
        .chunks_exact(2)
        .map(|pair| {
            circuit_pair_pauli_product(
                nqubits,
                axis,
                pair[0].0,
                pair[1].0,
                pair[0].1 != pair[1].1,
                instruction.line,
            )
        })
        .collect()
}

// ==============================================================================
// Argument validation
// ==============================================================================

fn check_probability(probability: f64, line: usize) -> Result<f64> {
    // Written as a negated containment so NaN is rejected.
    if !(0.0..=1.0).contains(&probability) {
        return Err(err_at(line, "probability must be between 0 and 1"));
    }
    Ok(probability)
}

fn ticit_paren_probability(instruction: &TicitInstruction) -> Result<f64> {
    if !instruction.has_parens {
        return Ok(0.0);
    }
    if instruction.parens.len() != 1 {
        return Err(err_at(
            instruction.line,
            "operation expects at most one probability argument",
        ));
    }
    check_probability(instruction.parens[0], instruction.line)
}

fn ticit_required_probability(instruction: &TicitInstruction) -> Result<f64> {
    if instruction.parens.len() != 1 {
        return Err(err_at(
            instruction.line,
            "operation expects one probability argument",
        ));
    }
    check_probability(instruction.parens[0], instruction.line)
}

fn require_paren_count(instruction: &TicitInstruction, count: usize, message: &str) -> Result<()> {
    if instruction.parens.len() != count {
        return Err(err_at(instruction.line, message));
    }
    Ok(())
}

fn require_no_parens(instruction: &TicitInstruction) -> Result<()> {
    if instruction.has_parens {
        return Err(err_at(
            instruction.line,
            "operation does not accept parens arguments",
        ));
    }
    Ok(())
}

fn require_no_targets(instruction: &TicitInstruction) -> Result<()> {
    if !instruction.targets.is_empty() {
        return Err(err_at(
            instruction.line,
            "operation does not accept targets",
        ));
    }
    Ok(())
}

fn require_coordinate_count(
    instruction: &TicitInstruction,
    parens_required: bool,
    message: &str,
) -> Result<()> {
    if instruction.has_parens {
        if instruction.parens.is_empty() || instruction.parens.len() > 16 {
            return Err(err_at(instruction.line, message));
        }
    } else if parens_required {
        return Err(err_at(instruction.line, message));
    }
    Ok(())
}

fn nonnegative_integer_argument(value: f64, message: &str, line: usize) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value.floor() != value || value > f64::from(i32::MAX) {
        return Err(err_at(line, message));
    }
    Ok(value as usize)
}

/// `I_ERROR`/`II_ERROR` are validated and then dropped: they document a noise
/// budget the simulator does not need to sample.
fn check_disjoint_probability_list(instruction: &TicitInstruction, message: &str) -> Result<()> {
    if instruction.has_parens && instruction.parens.is_empty() {
        return Err(err_at(instruction.line, message));
    }
    let mut total = 0.0;
    for &probability in &instruction.parens {
        total += check_probability(probability, instruction.line)?;
    }
    if total > 1.0 + 1e-12 {
        return Err(err_at(instruction.line, message));
    }
    Ok(())
}

// ==============================================================================
// Opcode tables
// ==============================================================================

fn single_qubit_clifford_kind(op: &str) -> Option<CircuitInstructionKind> {
    use CircuitInstructionKind as K;
    Some(match op {
        "H" | "H_XZ" => K::H,
        "H_NXY" => K::HNegXy,
        "H_NXZ" => K::HNegXz,
        "H_NYZ" => K::HNegYz,
        "H_XY" => K::HXy,
        "H_YZ" => K::HYz,
        "C_NXYZ" => K::CNegXyz,
        "C_NZYX" => K::CNegZyx,
        "C_XNYZ" => K::CXNegYz,
        "C_XYNZ" => K::CXyNegZ,
        "C_XYZ" => K::CXyz,
        "C_ZNYX" => K::CZNegYx,
        "C_ZYNX" => K::CZyNegX,
        "C_ZYX" => K::CZyx,
        "S" | "SQRT_Z" => K::S,
        "S_DAG" | "SQRT_Z_DAG" => K::SDag,
        "SQRT_X" => K::SqrtX,
        "SQRT_X_DAG" => K::SqrtXDag,
        "SQRT_Y" => K::SqrtY,
        "SQRT_Y_DAG" => K::SqrtYDag,
        "X" => K::X,
        "Y" => K::Y,
        "Z" => K::Z,
        _ => return None,
    })
}

fn two_qubit_clifford_kind(op: &str) -> Option<CircuitInstructionKind> {
    use CircuitInstructionKind as K;
    Some(match op {
        "CX" | "CNOT" | "ZCX" => K::CX,
        "CY" | "ZCY" => K::CY,
        "CZ" | "ZCZ" => K::CZ,
        "SWAP" => K::Swap,
        "CXSWAP" => K::CxSwap,
        "CZSWAP" | "SWAPCZ" => K::CzSwap,
        "ISWAP" => K::ISwap,
        "ISWAP_DAG" => K::ISwapDag,
        "SQRT_XX" => K::SqrtXx,
        "SQRT_XX_DAG" => K::SqrtXxDag,
        "SQRT_YY" => K::SqrtYy,
        "SQRT_YY_DAG" => K::SqrtYyDag,
        "SQRT_ZZ" => K::SqrtZz,
        "SQRT_ZZ_DAG" => K::SqrtZzDag,
        "SWAPCX" => K::SwapCx,
        "XCX" => K::Xcx,
        "XCY" => K::Xcy,
        "XCZ" => K::Xcz,
        "YCX" => K::Ycx,
        "YCY" => K::Ycy,
        "YCZ" => K::Ycz,
        _ => return None,
    })
}

// ==============================================================================
// Circuit assembly
// ==============================================================================

struct TicitCircuitBuilder {
    circuit: Circuit,
    coord_shift: Vec<f64>,
    correlated_error_products: Vec<CircuitPauliProduct>,
    correlated_error_probabilities: Vec<f64>,
    /// Probability that no alternative of the open correlated-error chain has
    /// fired yet. One when no chain is open, hence the hand-written `Default`.
    correlated_error_remaining_probability: f64,
}

impl Default for TicitCircuitBuilder {
    fn default() -> Self {
        Self {
            circuit: Circuit::default(),
            coord_shift: Vec::new(),
            correlated_error_products: Vec::new(),
            correlated_error_probabilities: Vec::new(),
            correlated_error_remaining_probability: 1.0,
        }
    }
}

impl TicitCircuitBuilder {
    fn push(&mut self, instruction: CircuitInstruction) {
        self.circuit.instructions.push(instruction);
    }
}

/// Ticit's `E`/`ELSE_CORRELATED_ERROR` chain is a sequence of *conditional*
/// alternatives; the IR wants absolute ones. `flush` closes the open chain and
/// emits it as a single categorical channel.
fn flush_correlated_error_group(builder: &mut TicitCircuitBuilder, line: usize) {
    builder.correlated_error_remaining_probability = 1.0;
    if builder.correlated_error_products.is_empty() {
        return;
    }
    let mut instruction =
        CircuitInstruction::new(CircuitInstructionKind::PauliProductChannel, line);
    instruction.pauli_products = std::mem::take(&mut builder.correlated_error_products);
    instruction.probabilities = std::mem::take(&mut builder.correlated_error_probabilities);
    builder.push(instruction);
}

fn append_correlated_error(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    starts_group: bool,
) -> Result<()> {
    if starts_group {
        flush_correlated_error_group(builder, instruction.line);
    } else if builder.correlated_error_products.is_empty() {
        return Err(err_at(
            instruction.line,
            "ELSE_CORRELATED_ERROR must follow CORRELATED_ERROR or ELSE_CORRELATED_ERROR",
        ));
    }
    let probability = ticit_required_probability(instruction)?;
    let product = circuit_implicit_pauli_product(builder.circuit.nqubits, instruction)?;
    builder.correlated_error_products.push(product);
    // p_i * prod_{j<i} (1 - p_j): the i-th alternative fires only if every
    // earlier one declined.
    let absolute = builder.correlated_error_remaining_probability * probability;
    builder.correlated_error_probabilities.push(absolute);
    builder.correlated_error_remaining_probability *= 1.0 - probability;
    Ok(())
}

fn circuit_measurement_targets(
    instruction: &TicitInstruction,
) -> Result<Vec<CircuitMeasurementTarget>> {
    Ok(ticit_measurement_targets(instruction)?
        .into_iter()
        .map(|(qubit, inverted)| CircuitMeasurementTarget { qubit, inverted })
        .collect())
}

fn append_qubit_instruction(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    kind: CircuitInstructionKind,
) -> Result<()> {
    let mut out = CircuitInstruction::new(kind, instruction.line);
    out.qubits = ticit_qubit_targets(instruction)?;
    builder.push(out);
    Ok(())
}

fn append_qubit_pair_instruction(
    builder: &mut TicitCircuitBuilder,
    line: usize,
    kind: CircuitInstructionKind,
    first: usize,
    second: usize,
) {
    let mut out = CircuitInstruction::new(kind, line);
    out.qubits = vec![first, second];
    builder.push(out);
}

fn append_measurement_instruction(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    kind: CircuitInstructionKind,
) -> Result<()> {
    let mut out = CircuitInstruction::new(kind, instruction.line);
    out.probability = ticit_paren_probability(instruction)?;
    out.measurement_targets = circuit_measurement_targets(instruction)?;
    builder.circuit.nrecords += out.measurement_targets.len();
    builder.push(out);
    Ok(())
}

fn append_probabilistic_qubit_instruction(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    kind: CircuitInstructionKind,
) -> Result<()> {
    let mut out = CircuitInstruction::new(kind, instruction.line);
    out.probability = ticit_required_probability(instruction)?;
    out.qubits = ticit_qubit_targets(instruction)?;
    builder.push(out);
    Ok(())
}

/// Ticit rotation arguments are half-turns; the kernel applies
/// `exp(-i * angle * P)`. `SPP`/`SPP_DAG` bypass this — they hardcode `±pi/4`.
fn pauli_rotation_kernel_angle_from_half_turns(half_turns: f64) -> f64 {
    half_turns * PI / 2.0
}

fn append_pauli_rotation_instruction(
    builder: &mut TicitCircuitBuilder,
    line: usize,
    kernel_angle: f64,
    products: Vec<CircuitPauliProduct>,
) {
    let mut out = CircuitInstruction::new(CircuitInstructionKind::PauliRotation, line);
    out.kernel_angle = kernel_angle;
    out.pauli_products = products;
    builder.push(out);
}

fn append_single_axis_rotation(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    axis: u8,
) -> Result<()> {
    require_paren_count(
        instruction,
        1,
        "single-qubit rotation expects one angle argument",
    )?;
    let nqubits = builder.circuit.nqubits;
    let products = ticit_qubit_targets(instruction)?
        .into_iter()
        .map(|qubit| circuit_single_pauli_product(nqubits, axis, qubit))
        .collect();
    append_pauli_rotation_instruction(
        builder,
        instruction.line,
        pauli_rotation_kernel_angle_from_half_turns(instruction.parens[0]),
        products,
    );
    Ok(())
}

fn append_two_axis_rotation(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    axis: u8,
) -> Result<()> {
    require_paren_count(
        instruction,
        1,
        "two-qubit Pauli rotation expects one angle argument",
    )?;
    let qubits = ticit_qubit_targets(instruction)?;
    if !qubits.len().is_multiple_of(2) {
        return Err(err_at(
            instruction.line,
            "two-qubit Pauli rotation requires paired targets",
        ));
    }
    let nqubits = builder.circuit.nqubits;
    let products = qubits
        .chunks_exact(2)
        .map(|pair| {
            circuit_pair_pauli_product(nqubits, axis, pair[0], pair[1], false, instruction.line)
        })
        .collect::<Result<Vec<_>>>()?;
    append_pauli_rotation_instruction(
        builder,
        instruction.line,
        pauli_rotation_kernel_angle_from_half_turns(instruction.parens[0]),
        products,
    );
    Ok(())
}

/// `U3(theta, phi, lambda)` decomposes as `Rz(phi) Ry(theta) Rz(lambda)`, so the
/// emitted order is Z(lambda), Y(theta), Z(phi) — arguments 2, 0, 1.
fn append_u3_rotation(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
) -> Result<()> {
    require_paren_count(instruction, 3, "U3 expects three angle arguments")?;
    let qubits = ticit_qubit_targets(instruction)?;
    let nqubits = builder.circuit.nqubits;
    for (argument, axis) in [(2usize, b'Z'), (0, b'Y'), (1, b'Z')] {
        let products = qubits
            .iter()
            .map(|&qubit| circuit_single_pauli_product(nqubits, axis, qubit))
            .collect();
        append_pauli_rotation_instruction(
            builder,
            instruction.line,
            pauli_rotation_kernel_angle_from_half_turns(instruction.parens[argument]),
            products,
        );
    }
    Ok(())
}

fn feedback_qubit_target(target: &str, line: usize) -> Result<usize> {
    let (qubit, inverted) = ticit_qubit_target(target, line)?;
    if inverted {
        return Err(err_at(
            line,
            "record-controlled feedback does not accept inverted qubit targets",
        ));
    }
    Ok(qubit)
}

fn append_feedback_pair(
    builder: &mut TicitCircuitBuilder,
    line: usize,
    kind: CircuitInstructionKind,
    record: usize,
    qubit: usize,
) {
    let mut out = CircuitInstruction::new(kind, line);
    out.feedback_targets
        .push(CircuitFeedbackTarget { record, qubit });
    builder.push(out);
}

fn ordinary_pair(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
    kind: CircuitInstructionKind,
    a: &str,
    b: &str,
) -> Result<()> {
    let (qa, inverted_a) = ticit_qubit_target(a, instruction.line)?;
    let (qb, inverted_b) = ticit_qubit_target(b, instruction.line)?;
    if inverted_a || inverted_b {
        return Err(err_at(
            instruction.line,
            "two-qubit Clifford does not accept inverted qubit targets",
        ));
    }
    append_qubit_pair_instruction(builder, instruction.line, kind, qa, qb);
    Ok(())
}

/// Handles the record-controlled forms of the two-qubit gates. Returns `false`
/// when the instruction has no `rec[-k]` target and should be dispatched as an
/// ordinary gate instead.
///
/// The pairing rules are asymmetric because Ticit's gate names encode which side
/// is the control: `CX`/`CY` take the record first, `XCZ`/`YCZ` second, and `CZ`
/// is symmetric so either side works.
fn append_classical_controlled_pairs(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
) -> Result<bool> {
    use CircuitInstructionKind as K;
    let op = instruction.op.as_str();
    let mut has_classical_target = false;
    for target in &instruction.targets {
        if is_sweep_target(target) {
            return Err(err_at(
                instruction.line,
                "sweep-controlled operations are not supported",
            ));
        }
        has_classical_target |= is_record_target(target);
    }
    if !has_classical_target {
        return Ok(false);
    }
    if !instruction.targets.len().is_multiple_of(2) {
        return Err(err_at(
            instruction.line,
            "controlled two-qubit gate requires paired targets",
        ));
    }
    for pair in instruction.targets.chunks_exact(2) {
        let (a, b) = (pair[0].as_str(), pair[1].as_str());
        let line = instruction.line;
        match op {
            "CX" | "CNOT" | "ZCX" | "CY" | "ZCY" => {
                let is_y = op == "CY" || op == "ZCY";
                if is_record_target(a) {
                    let qubit = feedback_qubit_target(b, line)?;
                    let record = ticit_record_index(a, builder.circuit.nrecords, line)?;
                    let kind = if is_y { K::FeedbackY } else { K::FeedbackX };
                    append_feedback_pair(builder, line, kind, record, qubit);
                } else {
                    let kind = if is_y { K::CY } else { K::CX };
                    ordinary_pair(builder, instruction, kind, a, b)?;
                }
            }
            "CZ" | "ZCZ" => {
                // CZ is symmetric, so the record may be on either side.
                let controlled = if is_record_target(a) {
                    Some((a, b))
                } else if is_record_target(b) {
                    Some((b, a))
                } else {
                    None
                };
                match controlled {
                    Some((record_target, qubit_target)) => {
                        let qubit = feedback_qubit_target(qubit_target, line)?;
                        let record =
                            ticit_record_index(record_target, builder.circuit.nrecords, line)?;
                        append_feedback_pair(builder, line, K::FeedbackZ, record, qubit);
                    }
                    None => ordinary_pair(builder, instruction, K::CZ, a, b)?,
                }
            }
            "XCZ" | "YCZ" => {
                let is_y = op == "YCZ";
                if is_record_target(b) {
                    let qubit = feedback_qubit_target(a, line)?;
                    let record = ticit_record_index(b, builder.circuit.nrecords, line)?;
                    let kind = if is_y { K::FeedbackY } else { K::FeedbackX };
                    append_feedback_pair(builder, line, kind, record, qubit);
                } else {
                    let kind = if is_y { K::Ycz } else { K::Xcz };
                    ordinary_pair(builder, instruction, kind, a, b)?;
                }
            }
            _ => return Err(err_at(line, "unsupported classically controlled gate")),
        }
    }
    Ok(true)
}

fn append_instruction(
    builder: &mut TicitCircuitBuilder,
    instruction: &TicitInstruction,
) -> Result<()> {
    use CircuitInstructionKind as K;
    let op = instruction.op.as_str();
    let line = instruction.line;

    // Correlated-error chains accumulate instead of emitting, so they are the
    // only ops that must not flush the open group first.
    if op == "E" || op == "CORRELATED_ERROR" {
        return append_correlated_error(builder, instruction, true);
    }
    if op == "ELSE_CORRELATED_ERROR" {
        return append_correlated_error(builder, instruction, false);
    }
    flush_correlated_error_group(builder, line);

    // --- Annotations, none of which emit an instruction except TICK ---------
    match op {
        "QUBIT_COORDS" => {
            require_coordinate_count(
                instruction,
                false,
                "QUBIT_COORDS expects 1 to 16 coordinate arguments when parens are present",
            )?;
            if instruction.targets.is_empty() {
                return Err(err_at(line, "QUBIT_COORDS expects qubit targets"));
            }
            // Validated for their side effect on the qubit count, then dropped.
            ticit_qubit_targets(instruction)?;
            return Ok(());
        }
        "TICK" => {
            require_no_parens(instruction)?;
            require_no_targets(instruction)?;
            builder.push(CircuitInstruction::new(K::Tick, line));
            return Ok(());
        }
        "SHIFT_COORDS" => {
            require_coordinate_count(
                instruction,
                true,
                "SHIFT_COORDS expects 1 to 16 coordinate arguments",
            )?;
            require_no_targets(instruction)?;
            if builder.coord_shift.len() < instruction.parens.len() {
                builder.coord_shift.resize(instruction.parens.len(), 0.0);
            }
            for (shift, offset) in builder.coord_shift.iter_mut().zip(&instruction.parens) {
                *shift += offset;
            }
            return Ok(());
        }
        "DETECTOR" | "DISCARD" => {
            require_coordinate_count(
                instruction,
                false,
                "detector expects 1 to 16 coordinate arguments when parens are present",
            )?;
            builder.circuit.detectors.push(CircuitDetector {
                records: ticit_record_indices(instruction, builder.circuit.nrecords)?,
                coords: coords_with_shift(&instruction.parens, &builder.coord_shift),
                line,
                after_instruction: builder.circuit.instructions.len(),
                discard: op == "DISCARD",
            });
            return Ok(());
        }
        "OBSERVABLE_INCLUDE" => {
            if instruction.parens.len() != 1 {
                return Err(err_at(
                    line,
                    "OBSERVABLE_INCLUDE expects one nonnegative observable index",
                ));
            }
            let index = nonnegative_integer_argument(
                instruction.parens[0],
                "OBSERVABLE_INCLUDE expects one nonnegative integer index",
                line,
            )?;
            builder.circuit.observables.push(CircuitObservableInclude {
                index,
                records: ticit_observable_record_indices(instruction, builder.circuit.nrecords)?,
                line,
            });
            return Ok(());
        }
        _ => {}
    }

    // Feedback dispatch runs before the gate tables: `CX rec[-1] 0` is a
    // classically controlled X, not a two-qubit Clifford. This is also the only
    // place sweep targets are detected, for any operation.
    if append_classical_controlled_pairs(builder, instruction)? {
        return Ok(());
    }

    let nqubits = builder.circuit.nqubits;
    if op == "I" {
        require_no_parens(instruction)?;
        ticit_qubit_targets(instruction)?;
    } else if op == "II" {
        require_no_parens(instruction)?;
        if !ticit_qubit_targets(instruction)?.len().is_multiple_of(2) {
            return Err(err_at(line, "II requires paired targets"));
        }
    } else if let Some(kind) = single_qubit_clifford_kind(op) {
        require_no_parens(instruction)?;
        append_qubit_instruction(builder, instruction, kind)?;
    } else if let Some(kind) = two_qubit_clifford_kind(op) {
        require_no_parens(instruction)?;
        append_qubit_instruction(builder, instruction, kind)?;
    } else if op == "T" {
        require_no_parens(instruction)?;
        append_qubit_instruction(builder, instruction, K::T)?;
    } else if op == "T_DAG" {
        require_no_parens(instruction)?;
        append_qubit_instruction(builder, instruction, K::TDag)?;
    } else if op == "R_X" {
        append_single_axis_rotation(builder, instruction, b'X')?;
    } else if op == "R_Y" {
        append_single_axis_rotation(builder, instruction, b'Y')?;
    } else if op == "R_Z" {
        append_single_axis_rotation(builder, instruction, b'Z')?;
    } else if op == "U3" || op == "U" {
        append_u3_rotation(builder, instruction)?;
    } else if op == "R_XX" {
        append_two_axis_rotation(builder, instruction, b'X')?;
    } else if op == "R_YY" {
        append_two_axis_rotation(builder, instruction, b'Y')?;
    } else if op == "R_ZZ" {
        append_two_axis_rotation(builder, instruction, b'Z')?;
    } else if op == "R_PAULI" {
        require_paren_count(instruction, 1, "R_PAULI expects one angle argument")?;
        let products = circuit_mpp_targets(nqubits, instruction)?;
        append_pauli_rotation_instruction(
            builder,
            line,
            pauli_rotation_kernel_angle_from_half_turns(instruction.parens[0]),
            products,
        );
    } else if matches!(op, "M" | "MZ" | "MX" | "MY") {
        let kind = match op {
            "MX" => K::MX,
            "MY" => K::MY,
            _ => K::MZ,
        };
        append_measurement_instruction(builder, instruction, kind)?;
    } else if matches!(op, "MR" | "MRZ" | "MRX" | "MRY") {
        let kind = match op {
            "MRX" => K::Mrx,
            "MRY" => K::Mry,
            _ => K::Mrz,
        };
        append_measurement_instruction(builder, instruction, kind)?;
    } else if matches!(op, "R" | "RZ" | "RX" | "RY") {
        require_no_parens(instruction)?;
        let kind = match op {
            "RX" => K::RX,
            "RY" => K::RY,
            _ => K::RZ,
        };
        append_qubit_instruction(builder, instruction, kind)?;
    } else if op == "MPP" {
        let mut out = CircuitInstruction::new(K::Mpp, line);
        out.probability = ticit_paren_probability(instruction)?;
        out.pauli_products = circuit_mpp_targets(nqubits, instruction)?;
        builder.circuit.nrecords += out.pauli_products.len();
        builder.push(out);
    } else if op == "EXP_VAL" {
        require_no_parens(instruction)?;
        let mut out = CircuitInstruction::new(K::ExpVal, line);
        out.pauli_products = circuit_mpp_targets(nqubits, instruction)?;
        if out.pauli_products.is_empty() {
            return Err(err_at(line, "EXP_VAL expects at least one Pauli product"));
        }
        out.exp_val = Some(builder.circuit.nexpvals);
        builder.circuit.nexpvals += out.pauli_products.len();
        builder.push(out);
    } else if matches!(op, "MXX" | "MYY" | "MZZ") {
        let mut out = CircuitInstruction::new(K::Mpp, line);
        out.probability = ticit_paren_probability(instruction)?;
        let axis = match op {
            "MXX" => b'X',
            "MYY" => b'Y',
            _ => b'Z',
        };
        out.pauli_products = circuit_pair_measurement_products(nqubits, instruction, axis)?;
        builder.circuit.nrecords += out.pauli_products.len();
        builder.push(out);
    } else if op == "SPP" || op == "SPP_DAG" {
        require_no_parens(instruction)?;
        let products = circuit_mpp_targets(nqubits, instruction)?;
        let kernel_angle = if op == "SPP" { PI / 4.0 } else { -PI / 4.0 };
        append_pauli_rotation_instruction(builder, line, kernel_angle, products);
    } else if matches!(op, "X_ERROR" | "Y_ERROR" | "Z_ERROR") {
        let kind = match op {
            "X_ERROR" => K::XError,
            "Y_ERROR" => K::YError,
            _ => K::ZError,
        };
        append_probabilistic_qubit_instruction(builder, instruction, kind)?;
    } else if op == "I_ERROR" {
        check_disjoint_probability_list(
            instruction,
            "I_ERROR probabilities must be disjoint and sum to at most 1",
        )?;
        ticit_qubit_targets(instruction)?;
    } else if op == "II_ERROR" {
        check_disjoint_probability_list(
            instruction,
            "II_ERROR probabilities must be disjoint and sum to at most 1",
        )?;
        if !ticit_qubit_targets(instruction)?.len().is_multiple_of(2) {
            return Err(err_at(line, "II_ERROR requires paired targets"));
        }
    } else if matches!(op, "DEPOLARIZE1" | "DEPOLARIZE2" | "DEPOLARIZE3") {
        let kind = match op {
            "DEPOLARIZE1" => K::Depolarize1,
            "DEPOLARIZE2" => K::Depolarize2,
            _ => K::Depolarize3,
        };
        append_probabilistic_qubit_instruction(builder, instruction, kind)?;
    } else if matches!(
        op,
        "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "PAULI_CHANNEL_3"
    ) {
        // One probability per non-identity Pauli on n qubits: 4^n - 1.
        let (expected, kind) = match op {
            "PAULI_CHANNEL_1" => (3, K::PauliChannel1),
            "PAULI_CHANNEL_2" => (15, K::PauliChannel2),
            _ => (63, K::PauliChannel3),
        };
        if instruction.parens.len() != expected {
            return Err(err_at(
                line,
                "PAULI_CHANNEL probability count does not match gate arity",
            ));
        }
        let mut out = CircuitInstruction::new(kind, line);
        out.probabilities = instruction.parens.clone();
        out.qubits = ticit_qubit_targets(instruction)?;
        builder.push(out);
    } else if op == "HERALDED_ERASE" {
        let mut out = CircuitInstruction::new(K::HeraldedErase, line);
        out.probability = ticit_required_probability(instruction)?;
        out.qubits = ticit_qubit_targets(instruction)?;
        // The herald itself is observed, one record per qubit.
        builder.circuit.nrecords += out.qubits.len();
        builder.push(out);
    } else if op == "HERALDED_PAULI_CHANNEL_1" {
        require_paren_count(
            instruction,
            4,
            "HERALDED_PAULI_CHANNEL_1 expects four probabilities",
        )?;
        let mut out = CircuitInstruction::new(K::HeraldedPauliChannel1, line);
        out.probabilities = instruction.parens.clone();
        out.qubits = ticit_qubit_targets(instruction)?;
        builder.circuit.nrecords += out.qubits.len();
        builder.push(out);
    } else if op == "MPAD" {
        let mut out = CircuitInstruction::new(K::MPad, line);
        out.probability = ticit_paren_probability(instruction)?;
        out.measurement_targets = circuit_measurement_targets(instruction)?;
        builder.circuit.nrecords += out.measurement_targets.len();
        builder.push(out);
    } else {
        return Err(err_at(line, format!("unsupported Ticit operation: {op}")));
    }
    Ok(())
}

fn coords_with_shift(coords: &[f64], shift: &[f64]) -> Vec<f64> {
    let width = coords.len().max(shift.len());
    (0..width)
        .map(|index| {
            coords.get(index).copied().unwrap_or(0.0) + shift.get(index).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Emits a node list, expanding `REPEAT` by literal repetition.
///
/// Each list ends with a correlated-error flush, so a chain can never straddle a
/// block boundary or a repetition boundary.
fn append_nodes(builder: &mut TicitCircuitBuilder, nodes: &[TicitNode]) -> Result<()> {
    for node in nodes {
        match node {
            TicitNode::Repeat(block) => {
                flush_correlated_error_group(builder, block.line);
                for _ in 0..block.count {
                    append_nodes(builder, &block.body)?;
                }
            }
            TicitNode::Instruction(instruction) => append_instruction(builder, instruction)?,
        }
    }
    let line = nodes.last().map_or(0, TicitNode::line);
    flush_correlated_error_group(builder, line);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_stripped_outside_brackets() {
        assert_eq!(strip_ticit_comment("  M 0  # measure  "), "M 0");
        assert_eq!(strip_ticit_comment("M[a#b] 0"), "M[a#b] 0");
        assert_eq!(strip_ticit_comment("# whole line"), "");
    }

    #[test]
    fn numeric_expressions_follow_precedence() {
        assert_eq!(parse_numeric_expression("1+2*3", 1).unwrap(), 7.0);
        assert_eq!(parse_numeric_expression("(1+2)*3", 1).unwrap(), 9.0);
        assert_eq!(parse_numeric_expression("-2", 1).unwrap(), -2.0);
        assert_eq!(parse_numeric_expression("--2", 1).unwrap(), 2.0);
        assert_eq!(parse_numeric_expression("pi/pi", 1).unwrap(), 1.0);
        assert_eq!(parse_numeric_expression("PI", 1).unwrap(), PI);
        assert_eq!(parse_numeric_expression("1e-3", 1).unwrap(), 1e-3);
    }

    #[test]
    fn unknown_constants_are_named_in_the_error() {
        let error = parse_numeric_expression("2*tau", 4).unwrap_err();
        assert!(error.message().contains("unknown numeric constant: TAU"));
        assert!(error.message().starts_with("line 4:"));
    }

    #[test]
    fn scan_number_stops_at_an_incomplete_exponent() {
        assert_eq!(scan_number(b"1e", 0), Some(1));
        assert_eq!(scan_number(b"1e5x", 0), Some(3));
        assert_eq!(scan_number(b".5+", 0), Some(2));
        assert_eq!(scan_number(b"x", 0), None);
    }

    #[test]
    fn record_targets_resolve_one_based() {
        assert!(is_record_target("rec[-1]"));
        assert!(!is_record_target("rec[1]"));
        assert!(!is_record_target("rec[-]"));
        assert_eq!(ticit_record_index("rec[-1]", 3, 1).unwrap(), 3);
        assert_eq!(ticit_record_index("rec[-3]", 3, 1).unwrap(), 1);
        assert!(ticit_record_index("rec[-4]", 3, 1).is_err());
        assert!(ticit_record_index("rec[-0]", 3, 1).is_err());
    }

    #[test]
    fn coordinate_shifts_widen_to_the_longer_vector() {
        assert_eq!(coords_with_shift(&[1.0], &[10.0, 20.0]), vec![11.0, 20.0]);
        assert_eq!(coords_with_shift(&[1.0, 2.0], &[]), vec![1.0, 2.0]);
        assert!(coords_with_shift(&[], &[]).is_empty());
    }
}

#[cfg(test)]
mod circuit_tests {
    //! `.ticit` parser structure, metadata, rejection, and corpus tests.

    use crate::test_support as common;

    use std::f64::consts::PI;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::circuit::ir::CircuitInstructionKind as Kind;

    // ==============================================================================
    // Helpers
    // ==============================================================================

    /// The C++ suite compares rotation angles to 1e-12; keep the same slack.
    const TOLERANCE: f64 = 1e-12;

    fn parse(text: &str) -> Circuit {
        parse_ticit_circuit_text(text)
            .unwrap_or_else(|error| panic!("{text:?} should parse: {error}"))
    }

    fn rejection_message(text: &str) -> String {
        parse_ticit_circuit_text(text)
            .expect_err(&format!("{text:?} should be rejected"))
            .message()
            .to_owned()
    }

    /// Asserts a rejection keeps the C++ message text *and* names its source line —
    /// the one deliberate improvement this port makes over the original.
    fn assert_rejected(text: &str, expected: &str) {
        let message = rejection_message(text);
        assert!(
            message.contains(expected),
            "expected {expected:?} in {message:?} for input {text:?}"
        );
        assert!(
            message.starts_with("line "),
            "missing line number in {message:?} for input {text:?}"
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    // ==============================================================================
    // Structure
    // ==============================================================================

    #[test]
    fn repeat_blocks_are_flattened_by_literal_repetition() {
        let circuit = parse("REPEAT 2 {\nM !0\n}\n");

        assert_eq!(circuit.instructions.len(), 2);
        assert_eq!(circuit.nrecords, 2);
        assert_eq!(circuit.nqubits, 1);
        for instruction in &circuit.instructions {
            assert_eq!(instruction.kind, Kind::MZ);
            assert_eq!(
                instruction.measurement_targets,
                vec![CircuitMeasurementTarget {
                    qubit: 0,
                    inverted: true,
                }]
            );
            // Every copy keeps the source line of the body, not of the block.
            assert_eq!(instruction.line, 2);
        }
    }

    #[test]
    fn empty_circuit_has_zero_metadata() {
        let circuit = parse("");

        assert_eq!(circuit.nqubits, 0);
        assert_eq!(circuit.nrecords, 0);
        assert_eq!(circuit.nexpvals, 0);
        assert_eq!(circuit.num_observables(), 0);
        assert!(circuit.instructions.is_empty());
        assert!(circuit.detectors.is_empty());
        assert!(circuit.observables.is_empty());
    }

    #[test]
    fn detectors_keep_record_order_and_stream_position() {
        let circuit = parse("M 0 1\nDETECTOR rec[-1] rec[-2]\nTICK\n");

        assert_eq!(circuit.detectors.len(), 1);
        let detector = &circuit.detectors[0];
        // Records are stored as written, not sorted: {rec[-1], rec[-2]} = {2, 1}.
        assert_eq!(detector.records, vec![2, 1]);
        assert_eq!(detector.line, 2);
        // One instruction (the M) precedes it; the TICK does not.
        assert_eq!(detector.after_instruction, 1);
        assert!(detector.coords.is_empty());
    }

    #[test]
    fn detector_metadata_matches_the_python_binding() {
        let circuit = parse("M 0\nDETECTOR(1.5, 2.5) rec[-1]\n");

        assert_eq!(circuit.detectors.len(), 1);
        assert_eq!(circuit.detectors[0].records, vec![1]);
        assert_eq!(circuit.detectors[0].coords, vec![1.5, 2.5]);
        assert_eq!(circuit.detectors[0].line, 2);
    }

    #[test]
    fn shift_coords_accumulate_into_detector_coordinates() {
        let circuit = parse(
            "M 0\n\
         SHIFT_COORDS(1, 2)\n\
         DETECTOR(0.5, 0.5) rec[-1]\n\
         SHIFT_COORDS(10)\n\
         DETECTOR(0.5, 0.5) rec[-1]\n\
         DETECTOR rec[-1]\n",
        );

        assert_eq!(circuit.detectors[0].coords, vec![1.5, 2.5]);
        assert_eq!(circuit.detectors[1].coords, vec![11.5, 2.5]);
        // A detector with no coordinates of its own still picks up the shift, and
        // widens to the shift's length.
        assert_eq!(circuit.detectors[2].coords, vec![11.0, 2.0]);
    }

    #[test]
    fn observable_count_is_the_largest_index_plus_one() {
        let circuit = parse("M 0\nOBSERVABLE_INCLUDE(2) rec[-1]\n");

        assert_eq!(circuit.num_observables(), 3);
        assert_eq!(circuit.observables.len(), 1);
        assert_eq!(circuit.observables[0].index, 2);
        assert_eq!(circuit.observables[0].records, vec![1]);
        assert_eq!(circuit.observables[0].line, 2);
    }

    #[test]
    fn several_includes_may_share_one_observable() {
        let circuit =
            parse("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n");

        assert_eq!(circuit.num_observables(), 1);
        assert_eq!(circuit.observables.len(), 2);
    }

    #[test]
    fn pair_measurement_mpad_and_observable_record_indexing() {
        let circuit = parse("MXX !0 1\nMPAD 1 0\nOBSERVABLE_INCLUDE(0) rec[-1] X0\n");

        assert_eq!(circuit.nrecords, 3);
        // The observable is an annotation; only MXX and MPAD emit instructions.
        assert_eq!(circuit.instructions.len(), 2);
        assert_eq!(circuit.instructions[0].kind, Kind::Mpp);
        assert_eq!(circuit.instructions[0].pauli_products.len(), 1);
        assert!(circuit.instructions[0].pauli_products[0].inverted);
        assert_eq!(circuit.instructions[1].kind, Kind::MPad);
        assert_eq!(circuit.observables.len(), 1);
        // The Pauli target is dropped; only rec[-1] survives, and it is the third
        // record because MPAD contributed two.
        assert_eq!(circuit.observables[0].records, vec![3]);
    }

    #[test]
    fn mpad_targets_are_literal_values_not_qubits() {
        let circuit = parse("MPAD 1\n");

        assert_eq!(circuit.nqubits, 0);
        assert_eq!(circuit.nrecords, 1);
        assert_eq!(circuit.instructions[0].kind, Kind::MPad);
        assert_eq!(circuit.instructions[0].measurement_targets[0].qubit, 1);
    }

    #[test]
    fn annotation_targets_do_not_grow_the_qubit_count() {
        let circuit = parse(
            "QUBIT_COORDS(0, 0) 7\n\
         M 0\n\
         DETECTOR rec[-1]\n\
         OBSERVABLE_INCLUDE(0) rec[-1] X99\n\
         MPAD 1\n\
         TICK\n\
         SHIFT_COORDS(1)\n",
        );

        // QUBIT_COORDS targets are real qubits; the other annotations' are not.
        assert_eq!(circuit.nqubits, 8);
    }

    #[test]
    fn exp_val_allocates_one_slot_per_product() {
        let circuit = parse("EXP_VAL Z0*Z1 X0*X1 X2\nEXP_VAL Z2\n");

        assert_eq!(circuit.nexpvals, 4);
        assert_eq!(circuit.nqubits, 3);
        assert_eq!(circuit.instructions[0].exp_val, Some(0));
        assert_eq!(circuit.instructions[0].pauli_products.len(), 3);
        assert_eq!(circuit.instructions[1].exp_val, Some(3));
        // EXP_VAL is a probe: it must not consume measurement records.
        assert_eq!(circuit.nrecords, 0);
    }

    #[test]
    fn operation_names_are_case_insensitive_and_tags_are_ignored() {
        let circuit = parse("h[gate-tag] 0\nm[another-tag] 0\n");

        assert_eq!(circuit.instructions[0].kind, Kind::H);
        assert_eq!(circuit.instructions[1].kind, Kind::MZ);
        assert_eq!(circuit.nrecords, 1);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored_but_still_count_for_line_numbers() {
        let circuit = parse("# header\n\nM 0  # trailing\n");

        assert_eq!(circuit.instructions.len(), 1);
        assert_eq!(circuit.instructions[0].line, 3);
    }

    // ==============================================================================
    // Angles
    // ==============================================================================

    #[test]
    fn single_qubit_rotation_arguments_are_half_turns() {
        let circuit = parse("R_X(0.5) 0\n");

        assert_eq!(circuit.instructions.len(), 1);
        assert_eq!(circuit.instructions[0].kind, Kind::PauliRotation);
        assert_close(circuit.instructions[0].kernel_angle, PI / 4.0);
        assert!(
            circuit.instructions[0].pauli_products[0]
                .pauli
                .same_body(&pauli_x(1, 0))
        );
    }

    #[test]
    fn rotation_angles_are_evaluated_as_expressions() {
        let circuit = parse("R_Z(pi/pi) 0\n");
        assert_close(circuit.instructions[0].kernel_angle, PI / 2.0);

        let circuit = parse("R_Z(2*PI - pi) 0\n");
        assert_close(circuit.instructions[0].kernel_angle, PI * PI / 2.0);
    }

    #[test]
    fn pauli_and_u3_rotations_use_half_turns() {
        let circuit = parse("R_XX(0.25) 0 1\nR_PAULI(-0.5) X0*Z1\nU3(0.5,0.25,-0.5) 0\n");

        assert_eq!(circuit.instructions.len(), 5);
        assert_close(circuit.instructions[0].kernel_angle, PI / 8.0);
        assert_close(circuit.instructions[1].kernel_angle, -PI / 4.0);
        // U3(theta, phi, lambda) emits Z(lambda), Y(theta), Z(phi).
        assert_close(circuit.instructions[2].kernel_angle, -PI / 4.0);
        assert_close(circuit.instructions[3].kernel_angle, PI / 4.0);
        assert_close(circuit.instructions[4].kernel_angle, PI / 8.0);

        let axis_of = |index: usize| {
            let pauli = &circuit.instructions[index].pauli_products[0].pauli;
            (pauli.xbit(0), pauli.zbit(0))
        };
        assert_eq!(axis_of(2), (false, true));
        assert_eq!(axis_of(3), (true, true));
        assert_eq!(axis_of(4), (false, true));
    }

    #[test]
    fn spp_bypasses_the_half_turn_conversion() {
        let circuit = parse("SPP X0*Z1\nSPP_DAG X0*Z1\n");

        assert_close(circuit.instructions[0].kernel_angle, PI / 4.0);
        assert_close(circuit.instructions[1].kernel_angle, -PI / 4.0);
    }

    // ==============================================================================
    // Channels
    // ==============================================================================

    #[test]
    fn correlated_error_chain_becomes_one_absolute_channel() {
        let circuit = parse("E(0.25) X0\nELSE_CORRELATED_ERROR(0.5) Z0\nM 0\n");

        assert_eq!(circuit.instructions.len(), 2);
        assert_eq!(circuit.instructions[0].kind, Kind::PauliProductChannel);
        assert_eq!(circuit.instructions[0].pauli_products.len(), 2);
        assert_eq!(circuit.instructions[0].probabilities.len(), 2);
        assert_close(circuit.instructions[0].probabilities[0], 0.25);
        // 0.375 = 0.5 * (1 - 0.25): the ELSE branch is conditional in the source
        // and absolute in the IR.
        assert_close(circuit.instructions[0].probabilities[1], 0.375);
    }

    #[test]
    fn correlated_error_chains_flush_at_input_end_and_repeat_boundaries() {
        let circuit = parse("E(0.25) X0\n");
        assert_eq!(circuit.instructions.len(), 1);
        assert_eq!(circuit.instructions[0].kind, Kind::PauliProductChannel);

        let circuit = parse("E(0.25) X0\nREPEAT 2 {\nE(0.5) Z0\n}\n");
        assert_eq!(circuit.instructions.len(), 3);
        for instruction in &circuit.instructions {
            assert_eq!(instruction.kind, Kind::PauliProductChannel);
            assert_eq!(instruction.probabilities.len(), 1);
        }
        assert_close(circuit.instructions[0].probabilities[0], 0.25);
        // A chain cannot straddle a repetition, so each copy restarts at p.
        assert_close(circuit.instructions[1].probabilities[0], 0.5);
        assert_close(circuit.instructions[2].probabilities[0], 0.5);
    }

    #[test]
    fn heralded_channels_append_one_record_per_qubit() {
        let circuit = parse(
            "SPP X0*Z1\n\
         PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0\n\
         PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1) 0 1\n\
         DEPOLARIZE3(0.1) 0 1 2\n\
         HERALDED_ERASE(0.1) 0\n\
         HERALDED_PAULI_CHANNEL_1(0, 0.1, 0, 0) 1\n\
         M 0 1\n",
        );

        // Two heralds ahead of the two M records.
        assert_eq!(circuit.nrecords, 4);
        assert_eq!(circuit.nqubits, 3);
        let kinds: Vec<_> = circuit.instructions.iter().map(|i| i.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::PauliRotation,
                Kind::PauliChannel1,
                Kind::PauliChannel2,
                Kind::Depolarize3,
                Kind::HeraldedErase,
                Kind::HeraldedPauliChannel1,
                Kind::MZ,
            ]
        );
        assert_eq!(circuit.instructions[1].probabilities, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn identity_errors_are_validated_and_dropped() {
        let circuit = parse("I_ERROR(0.5, 0.25) 0\nII_ERROR(0.5) 0 1\nI 2\nII 3 4\n");

        assert!(circuit.instructions.is_empty());
        // Their targets still count as qubits.
        assert_eq!(circuit.nqubits, 5);
    }

    // ==============================================================================
    // Feedback and combiners
    // ==============================================================================

    #[test]
    fn feedback_pairing_follows_the_gate_name() {
        let circuit = parse(
            "M 0\n\
         CX rec[-1] 1\n\
         CY rec[-1] 1\n\
         CZ rec[-1] 1\n\
         CZ 1 rec[-1]\n\
         XCZ 1 rec[-1]\n\
         YCZ 1 rec[-1]\n",
        );

        let kinds: Vec<_> = circuit.instructions[1..].iter().map(|i| i.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::FeedbackX,
                Kind::FeedbackY,
                Kind::FeedbackZ,
                Kind::FeedbackZ,
                Kind::FeedbackX,
                Kind::FeedbackY,
            ]
        );
        for instruction in &circuit.instructions[1..] {
            assert_eq!(
                instruction.feedback_targets,
                vec![CircuitFeedbackTarget {
                    record: 1,
                    qubit: 1,
                }]
            );
        }
    }

    #[test]
    fn pairs_without_a_record_fall_back_to_the_ordinary_gate() {
        let circuit = parse("M 0\nCX rec[-1] 1 2 3\n");

        assert_eq!(circuit.instructions[1].kind, Kind::FeedbackX);
        assert_eq!(circuit.instructions[2].kind, Kind::CX);
        assert_eq!(circuit.instructions[2].qubits, vec![2, 3]);
    }

    #[test]
    fn mpp_combiners_regroup_across_whitespace() {
        let spaced = parse("MPP X0 * X1 Z2\n");
        let joined = parse("MPP X0*X1 Z2\n");
        let half_joined = parse("MPP X0* X1 Z2\n");

        assert_eq!(spaced.nrecords, 2);
        assert_eq!(spaced.instructions[0].pauli_products.len(), 2);
        assert_eq!(
            spaced.instructions[0].pauli_products,
            joined.instructions[0].pauli_products
        );
        assert_eq!(
            spaced.instructions[0].pauli_products,
            half_joined.instructions[0].pauli_products
        );
    }

    #[test]
    fn each_inversion_toggles_the_whole_product_sign() {
        assert!(!parse("MPP !X0*!Z1\n").instructions[0].pauli_products[0].inverted);
        assert!(parse("MPP !X0*Z1\n").instructions[0].pauli_products[0].inverted);
        // MXX takes the XOR of the pair's flags, not of each factor's.
        assert!(parse("MXX !0 1\n").instructions[0].pauli_products[0].inverted);
        assert!(!parse("MXX !0 !1\n").instructions[0].pauli_products[0].inverted);
    }

    #[test]
    fn aliases_map_onto_their_canonical_kinds() {
        let circuit = parse(
            "H_XZ 0\nSQRT_Z 0\nSQRT_Z_DAG 0\nZCX 0 1\nCNOT 0 1\nZCY 0 1\nZCZ 0 1\nSWAPCZ 0 1\n\
         MZ 0\nMRZ 0\nRZ 0\nU(0.5,0.25,-0.5) 0\n",
        );

        let kinds: Vec<_> = circuit.instructions.iter().map(|i| i.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::H,
                Kind::S,
                Kind::SDag,
                Kind::CX,
                Kind::CX,
                Kind::CY,
                Kind::CZ,
                Kind::CzSwap,
                Kind::MZ,
                Kind::Mrz,
                Kind::RZ,
                // `U` is the three-rotation `U3` alias.
                Kind::PauliRotation,
                Kind::PauliRotation,
                Kind::PauliRotation,
            ]
        );
    }

    // ==============================================================================
    // Rejections
    // ==============================================================================

    #[test]
    fn errors_name_the_offending_line() {
        assert_eq!(
            rejection_message("H 0\nH 1\nFOO 2\n"),
            "line 3: unsupported Ticit operation: FOO"
        );
    }

    /// Every rejection pinned by the C++ suite (`§4.3(c)` of the test catalogue),
    /// plus the messages the C++ can produce that the suite never exercised.
    #[test]
    fn parse_errors_keep_the_cpp_message_text() {
        let cases: &[(&str, &str)] = &[
            // -- catalogue §4.3(c) --------------------------------------------
            (
                "X 0\nCX sweep[0] 0\n",
                "sweep-controlled operations are not supported",
            ),
            (
                "X 0\nCZ 0 sweep[0]\n",
                "sweep-controlled operations are not supported",
            ),
            ("X 0\nCX sweep[x] 0\n", "invalid qubit target"),
            ("REPEAT 0 {\nM 0\n}\n", "REPEAT count must be in [1, 10^18]"),
            ("REPEAT(2) 1 {\nM 0\n}\n", "invalid Ticit REPEAT block"),
            (
                "M 0\nOBSERVABLE_INCLUDE(0.6) rec[-1]\n",
                "OBSERVABLE_INCLUDE expects one nonnegative integer index",
            ),
            ("R(0.25) 0\n", "operation does not accept parens arguments"),
            (
                "I_ERROR(0.8,0.8) 0\n",
                "I_ERROR probabilities must be disjoint and sum to at most 1",
            ),
            (
                "II_ERROR(0.8,0.8) 0 1\n",
                "II_ERROR probabilities must be disjoint and sum to at most 1",
            ),
            ("TICK 0\n", "operation does not accept targets"),
            ("TICK()\n", "operation does not accept parens arguments"),
            ("SHIFT_COORDS(1) 0\n", "operation does not accept targets"),
            (
                "SHIFT_COORDS\n",
                "SHIFT_COORDS expects 1 to 16 coordinate arguments",
            ),
            ("QUBIT_COORDS foo\n", "invalid qubit target"),
            (
                "QUBIT_COORDS() 0\n",
                "QUBIT_COORDS expects 1 to 16 coordinate arguments when parens are present",
            ),
            // -- block structure ----------------------------------------------
            ("FOO 0\n", "unsupported Ticit operation: FOO"),
            ("M 0\n}\n", "unmatched Ticit block terminator"),
            ("REPEAT 2 {\nM 0\n", "unterminated Ticit block"),
            ("REPEAT 2\n", "REPEAT must start a block"),
            (
                "REPEAT 3000000000 {\nM 0\n}\n",
                "REPEAT count is too large for this flattened circuit frontend",
            ),
            (
                "REPEAT 10000000000000000000 {\nM 0\n}\n",
                "REPEAT count must be in [1, 10^18]",
            ),
            ("0 1\n", "invalid Ticit instruction"),
            ("M[unterminated 0\n", "unterminated Ticit tag"),
            // -- numeric arguments --------------------------------------------
            ("R_Z(tau) 0\n", "unknown numeric constant: TAU"),
            ("R_Z(1 2) 0\n", "invalid numeric expression"),
            ("R_Z((1) 0\n", "unterminated Ticit argument list"),
            ("R_Z(1,) 0\n", "invalid numeric expression"),
            ("X_ERROR(1.5) 0\n", "probability must be between 0 and 1"),
            ("X_ERROR(nan) 0\n", "probability must be between 0 and 1"),
            ("X_ERROR 0\n", "operation expects one probability argument"),
            (
                "M(0.1, 0.2) 0\n",
                "operation expects at most one probability argument",
            ),
            (
                "R_X(1,2) 0\n",
                "single-qubit rotation expects one angle argument",
            ),
            ("U3(0.5) 0\n", "U3 expects three angle arguments"),
            ("R_PAULI 0\n", "R_PAULI expects one angle argument"),
            (
                "PAULI_CHANNEL_1(0.1, 0.2) 0\n",
                "PAULI_CHANNEL probability count does not match gate arity",
            ),
            (
                "HERALDED_PAULI_CHANNEL_1(0.1) 0\n",
                "HERALDED_PAULI_CHANNEL_1 expects four probabilities",
            ),
            (
                "SPP(0.5) X0\n",
                "operation does not accept parens arguments",
            ),
            // -- targets -------------------------------------------------------
            ("H !0\n", "operation does not accept inverted targets"),
            ("MPP * X0\n", "misplaced MPP combiner"),
            ("MPP X0 *\n", "dangling MPP combiner"),
            ("MPP X0*Q1\n", "invalid MPP factor"),
            ("MPP X\n", "invalid MPP factor"),
            ("EXP_VAL\n", "EXP_VAL expects at least one Pauli product"),
            (
                "M 0\nDETECTOR rec[-2]\n",
                "measurement record target out of range",
            ),
            ("DETECTOR 0\n", "invalid measurement record target"),
            (
                "M 0\nOBSERVABLE_INCLUDE rec[-1]\n",
                "OBSERVABLE_INCLUDE expects one nonnegative observable index",
            ),
            (
                "M 0\nOBSERVABLE_INCLUDE(0) foo\n",
                "OBSERVABLE_INCLUDE targets must be measurement records or Pauli targets",
            ),
            (
                "MZZ 0 0\n",
                "two-qubit Pauli operation requires distinct qubits",
            ),
            (
                "R_XX(0.25) 0 0\n",
                "two-qubit Pauli operation requires distinct qubits",
            ),
            (
                "R_XX(0.25) 0 1 2\n",
                "two-qubit Pauli rotation requires paired targets",
            ),
            ("MZZ 0 1 2\n", "pair measurement requires paired targets"),
            ("II 0 1 2\n", "II requires paired targets"),
            ("II_ERROR(0.1) 0 1 2\n", "II_ERROR requires paired targets"),
            // -- feedback ------------------------------------------------------
            (
                "M 0\nH rec[-1] 0\n",
                "unsupported classically controlled gate",
            ),
            (
                "M 0\nCX rec[-1]\n",
                "controlled two-qubit gate requires paired targets",
            ),
            (
                "M 0\nCX rec[-1] !1\n",
                "record-controlled feedback does not accept inverted qubit targets",
            ),
            (
                "M 0\nCX rec[-1] 1 !2 3\n",
                "two-qubit Clifford does not accept inverted qubit targets",
            ),
            // -- correlated errors ---------------------------------------------
            (
                "ELSE_CORRELATED_ERROR(0.5) X0\n",
                "ELSE_CORRELATED_ERROR must follow CORRELATED_ERROR or ELSE_CORRELATED_ERROR",
            ),
            (
                "E(0.25) X0 * Z1\n",
                "implicit Pauli products do not use combiner targets",
            ),
            ("E X0\n", "operation expects one probability argument"),
        ];

        for (text, expected) in cases {
            assert_rejected(text, expected);
        }
    }

    #[test]
    fn coordinate_lists_are_capped_at_sixteen() {
        let seventeen = (0..17).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        assert_rejected(
            &format!("M 0\nDETECTOR({seventeen}) rec[-1]\n"),
            "detector expects 1 to 16 coordinate arguments when parens are present",
        );
        assert_rejected(
            &format!("SHIFT_COORDS({seventeen})\n"),
            "SHIFT_COORDS expects 1 to 16 coordinate arguments",
        );
    }

    /// The three places where this port deliberately diverges, all documented in
    /// `src/stim_parser.rs`. A differential run against the C++ frontend over 197
    /// corner cases and both benchmark corpora found no other difference.
    #[test]
    fn deliberate_deviations_from_the_cpp_frontend() {
        // The C++ counts qubits with an all-digits test but resolves factors with
        // `std::stoi`, so these slip past the count and then fail deep inside Pauli
        // construction with "qubit index out of range".
        assert_rejected("MPP X0abc\n", "invalid MPP factor");
        assert_rejected("MPP X+5\n", "invalid MPP factor");
        // The C++ reaches this through `strtod`, which accepts hexadecimal floats.
        assert_rejected("R_Z(0x10) 0\n", "invalid numeric expression");
    }

    #[test]
    fn a_missing_file_is_reported_by_path() {
        let error = parse_ticit_circuit_file("/nonexistent/circuit.stim")
            .expect_err("a missing file should be rejected");
        assert!(
            error
                .message()
                .contains("failed to read circuit file: /nonexistent/circuit.stim"),
            "{error}"
        );
    }

    // ==============================================================================
    // Benchmark corpora
    // ==============================================================================

    /// Lists a corpus, or returns empty when the corpus is not present on this
    /// machine. Both corpora live outside the repository, so their absence must not
    /// fail the suite.
    fn corpus(directory: &Path, extension: &str) -> Vec<PathBuf> {
        let entries = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()));
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|value| value == extension))
            .collect();
        paths.sort();
        paths
    }

    fn parse_corpus(directory: &Path, extension: &str, minimum: usize) {
        let paths = corpus(directory, extension);
        assert!(
            paths.len() >= minimum,
            "expected at least {minimum} circuits in {}, found {}",
            directory.display(),
            paths.len()
        );
        for path in &paths {
            parse_ticit_circuit_file(path)
                .unwrap_or_else(|error| panic!("{} failed to parse: {error}", path.display()));
        }
    }

    #[test]
    fn every_soft_benchmark_circuit_parses() {
        let directory = common::soft_benchmark_circuits();
        parse_corpus(&directory, "stim", 9);
        // Spot-check against benchmark/circuit/manifest.json, which records
        // `qubits` and `measurements` per circuit.
        let path = directory.join("msc_d3_inject_cultivate_p1e-3.stim");
        let circuit = parse_ticit_circuit_file(&path).expect("manifest circuit parses");
        assert_eq!(circuit.nqubits, 15);
        assert_eq!(circuit.nrecords, 21);
        assert_eq!(circuit.detectors.len(), 20);
        assert_eq!(circuit.num_observables(), 1);
    }

    #[test]
    fn every_ccz_nontels_circuit_parses() {
        let directory = common::ccz_nontels_circuits();
        parse_corpus(&directory, "clifft", 8);
        // `.clifft` is plain Ticit dialect. Counts derived independently from the
        // file text; the bundle ships no metadata of its own.
        let path = directory.join("d05_p0.clifft");
        let circuit = parse_ticit_circuit_file(&path).expect("ccz bundle circuit parses");
        assert_eq!(circuit.nqubits, 835);
        assert_eq!(circuit.nrecords, 8220);
        assert_eq!(circuit.nexpvals, 100);
        // The bundle is detector-free by construction: raw records and EXP_VAL only.
        assert!(circuit.detectors.is_empty());
        assert!(circuit.observables.is_empty());
    }
}
