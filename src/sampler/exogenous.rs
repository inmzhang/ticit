//! Presampling of exogenous symbols — noise coins that do not depend
//! on anything the circuit computes.
//!
//! The symbol-major [`PackedPresampledExogenous`] gives each condition a
//! contiguous bit-plane across shots for the batch/expression paths.

use crate::bits::packed_bit;
use crate::bits::{check_probability, symbol_bit_mask, symbol_word_count, symbol_word_index};
use crate::errors::{Result, TicitError};
use crate::factored::{
    BernoulliSampleGroup, FactoredInstructionProgram, RareCategoricalSampleGroup,
};
use crate::planner::LOW_PROBABILITY_SAMPLE_THRESHOLD;
use crate::random::{
    geometric_gap_denominator, next_random_u64, sample_categorical_row,
    sample_geometric_gap_with_denominator,
};
use crate::symbolic::SymbolicCategoricalDistribution;

/// Symbol-major packed table:
/// `value_words[(condition - 1) * shot_words + shot_word]`.
///
/// Conditions sampled by the geometric-skip families (rare categorical and
/// low-probability Bernoulli groups) set only a handful of bits per row, so
/// their rows are not materialized at all: each hit is recorded as a
/// `(condition, shot)` pair and exposed through a per-condition CSR index
/// with XOR semantics. Their `value_words` rows are unspecified; consumers
/// must check [`Self::is_sparse_condition`] first.
#[derive(Clone, Debug, Default)]
pub struct PackedPresampledExogenous {
    pub nshots: usize,
    pub nsymbols: usize,
    pub shot_words: usize,
    pub next_rng_state: u64,
    pub exogenous_assigned_words: Vec<u64>,
    pub value_words: Vec<u64>,
    /// Bitset over `symbol_word_index`/`symbol_bit_mask`: condition rows kept
    /// sparse in the *current* resample. Starts from the template each chunk;
    /// groups whose measured hits exceed the density threshold are cleared
    /// here and materialized densely instead.
    pub sparse_condition_words: Vec<u64>,
    /// Prepare-time template: every skip-family condition.
    pub sparse_condition_template_words: Vec<u64>,
    /// CSR over conditions: condition `c`'s hit shots live at
    /// `sparse_hit_shots[sparse_hit_offsets[c - 1]..sparse_hit_offsets[c]]`.
    pub sparse_hit_offsets: Vec<u32>,
    /// Hit shot indices, XOR semantics per hit (duplicates toggle back off).
    pub sparse_hit_shots: Vec<u32>,
    /// Draw-order `(condition - 1, shot)` scratch reused between resamples.
    pub scratch_hits: Vec<(u32, u32)>,
    /// Counting-sort cursor scratch reused between resamples.
    pub scratch_cursors: Vec<u32>,
}

impl PackedPresampledExogenous {
    pub fn is_sparse_condition(&self, condition: i32) -> bool {
        let word = symbol_word_index(condition);
        word < self.sparse_condition_words.len()
            && self.sparse_condition_words[word] & symbol_bit_mask(condition) != 0
    }

    /// Hit shots for a sparse condition in this resample's chunk.
    pub fn sparse_condition_hits(&self, condition: i32) -> &[u32] {
        let index = (condition - 1) as usize;
        let start = self.sparse_hit_offsets[index] as usize;
        let end = self.sparse_hit_offsets[index + 1] as usize;
        &self.sparse_hit_shots[start..end]
    }
}

/// Bitset of every condition the packed presampler keeps sparse: the two
/// geometric-skip scatter families.
fn sparse_condition_words(program: &FactoredInstructionProgram) -> Result<Vec<u64>> {
    let mut words = vec![0u64; symbol_word_count(program.nsymbols)];
    for group in &program.sampled_rare_categorical_groups {
        for conditions in &group.conditions {
            mark_exogenous_conditions(&mut words, conditions)?;
        }
    }
    for group in &program.sampled_low_probability_bernoulli_groups {
        mark_exogenous_conditions(&mut words, &group.conditions)?;
    }
    Ok(words)
}

/// Counting-sorts the draw-order hit scratch into the CSR index.
fn build_sparse_condition_hits(
    nsymbols: usize,
    scratch_hits: &[(u32, u32)],
    scratch_cursors: &mut Vec<u32>,
    sparse_hit_offsets: &mut Vec<u32>,
    sparse_hit_shots: &mut Vec<u32>,
) {
    sparse_hit_offsets.clear();
    sparse_hit_offsets.resize(nsymbols + 1, 0);
    for &(row, _) in scratch_hits {
        sparse_hit_offsets[row as usize + 1] += 1;
    }
    for index in 1..=nsymbols {
        sparse_hit_offsets[index] += sparse_hit_offsets[index - 1];
    }
    scratch_cursors.clear();
    scratch_cursors.extend_from_slice(&sparse_hit_offsets[..nsymbols]);
    sparse_hit_shots.clear();
    sparse_hit_shots.resize(scratch_hits.len(), 0);
    for &(row, shot) in scratch_hits {
        let cursor = &mut scratch_cursors[row as usize];
        sparse_hit_shots[*cursor as usize] = shot;
        *cursor += 1;
    }
}

fn packed_shot_word_count(shots: usize) -> usize {
    shots.div_ceil(64)
}

fn low_bits_mask(nbits: i64) -> u64 {
    if nbits <= 0 {
        return 0;
    }
    if nbits >= 64 {
        return u64::MAX;
    }
    (1u64 << nbits) - 1
}

fn packed_live_word_mask(shots: usize, word: usize) -> u64 {
    low_bits_mask(shots as i64 - ((word as i64) << 6))
}

fn packed_condition_offset(shot_words: usize, condition: i32, shot_word: usize) -> usize {
    (condition - 1) as usize * shot_words + shot_word
}

fn xor_packed_presampled_condition(
    value_words: &mut [u64],
    shot_words: usize,
    shot: usize,
    condition: i32,
) {
    if shot_words == 0 {
        return;
    }
    let word = shot >> 6;
    let mask = 1u64 << (shot & 63);
    value_words[packed_condition_offset(shot_words, condition, word)] ^= mask;
}

/// Sparse OR of Bernoulli(p) hits into one packed row via geometric skips.
fn or_low_probability_bits_packed(
    row: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    probability: f64,
    shots: usize,
) -> Result<()> {
    if probability <= 0.0 || shots == 0 {
        return Ok(());
    }
    let gap_denominator = geometric_gap_denominator(probability)?;
    let mut draw = 0usize;
    loop {
        let gap = sample_geometric_gap_with_denominator(rng_state, gap_denominator) as usize;
        if gap >= shots - draw {
            return Ok(());
        }
        draw += gap;
        let word = draw >> 6;
        if word >= shot_words {
            return Ok(());
        }
        row[word] |= 1u64 << (draw & 63);
        draw += 1;
    }
}

/// Fills one packed row with Bernoulli(p) bits.
///
/// The mid-range branch truncates `p` to 8 binary digits and generates those
/// with fair coin words (exactly 8 draws per shot word), then adds the
/// leftover mass with a geometric pass; `p > 0.5` samples the complement and
/// inverts. This is one of the three deliberately different Bernoulli
/// generators (the others are per-shot draws and the geometric skip) and its
/// draw pattern is part of the output contract.
fn generate_packed_biased_bits(
    row: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    probability: f64,
    shots: usize,
) -> Result<()> {
    if shots == 0 || shot_words == 0 || probability <= 0.0 {
        return Ok(());
    }
    if probability >= 1.0 {
        for (word, slot) in row.iter_mut().enumerate().take(shot_words) {
            *slot = packed_live_word_mask(shots, word);
        }
        return Ok(());
    }

    let invert = probability > 0.5;
    let p = if invert {
        1.0 - probability
    } else {
        probability
    };
    if p <= 0.0 {
        for (word, slot) in row.iter_mut().enumerate().take(shot_words) {
            *slot = if invert {
                packed_live_word_mask(shots, word)
            } else {
                0
            };
        }
        return Ok(());
    }
    if p == 0.5 {
        for (word, slot) in row.iter_mut().enumerate().take(shot_words) {
            *slot = next_random_u64(rng_state) & packed_live_word_mask(shots, word);
        }
    } else if p < LOW_PROBABILITY_SAMPLE_THRESHOLD {
        or_low_probability_bits_packed(row, shot_words, rng_state, p, shots)?;
    } else {
        const COIN_FLIPS: u32 = 8;
        let buckets = (1u64 << COIN_FLIPS) as f64;
        let raw_top_bits = (p * buckets) as u64;
        let half = 1u64 << (COIN_FLIPS - 1);
        let top_bits = if raw_top_bits < half {
            raw_top_bits
        } else {
            half - 1
        };
        let p_truncated = top_bits as f64 / buckets;
        for (word, slot) in row.iter_mut().enumerate().take(shot_words) {
            let mut alive = next_random_u64(rng_state);
            let mut result = 0u64;
            for bit in (0..=(COIN_FLIPS - 2)).rev() {
                let shoot = next_random_u64(rng_state);
                if (top_bits >> bit) & 1 != 0 {
                    result |= shoot & alive;
                }
                alive &= !shoot;
            }
            *slot = result & packed_live_word_mask(shots, word);
        }

        let p_leftover = p - p_truncated;
        if p_leftover > 0.0 {
            or_low_probability_bits_packed(
                row,
                shot_words,
                rng_state,
                p_leftover / (1.0 - p_truncated),
                shots,
            )?;
        }
    }

    if invert {
        for (word, slot) in row.iter_mut().enumerate().take(shot_words) {
            *slot = !*slot & packed_live_word_mask(shots, word);
        }
    }
    Ok(())
}

fn mark_exogenous_conditions(words: &mut [u64], conditions: &[i32]) -> Result<()> {
    for &condition in conditions {
        let word = symbol_word_index(condition);
        if word >= words.len() {
            return Err(TicitError::new(
                "exogenous condition exceeds program symbol table",
            ));
        }
        words[word] |= symbol_bit_mask(condition);
    }
    Ok(())
}

/// Bitset of every condition the presampler will assign, i.e. the split key
/// between exogenous and residual conditions in expression plans.
pub fn exogenous_assigned_words(program: &FactoredInstructionProgram) -> Result<Vec<u64>> {
    let mut words = vec![0u64; symbol_word_count(program.nsymbols)];
    for distribution in &program.sampled_categorical_distributions {
        mark_exogenous_conditions(&mut words, &distribution.conditions)?;
    }
    for group in &program.sampled_rare_categorical_groups {
        for conditions in &group.conditions {
            mark_exogenous_conditions(&mut words, conditions)?;
        }
    }
    mark_exogenous_conditions(&mut words, &program.sampled_bernoulli_conditions)?;
    for group in &program.sampled_low_probability_bernoulli_groups {
        mark_exogenous_conditions(&mut words, &group.conditions)?;
    }
    Ok(words)
}

// ==============================================================================
// Packed (symbol-major) presampling
// ==============================================================================

fn presample_categorical_distribution_packed(
    value_words: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    distribution: &SymbolicCategoricalDistribution,
    shots: usize,
) {
    for shot in 0..shots {
        let row = sample_categorical_row(rng_state, &distribution.probabilities);
        let assignment = &distribution.assignments[row];
        for (bit_idx, &condition) in distribution.conditions.iter().enumerate() {
            if packed_bit(assignment, bit_idx) {
                xor_packed_presampled_condition(value_words, shot_words, shot, condition);
            }
        }
    }
}

fn presample_rare_categorical_group_packed_dense(
    value_words: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    group: &RareCategoricalSampleGroup,
    shots: usize,
) -> Result<()> {
    let nsets = group.conditions.len();
    if group.event_probability <= 0.0 || nsets == 0 {
        return Ok(());
    }
    let total_draws = (shots as i64) * (nsets as i64);
    let gap_denominator = geometric_gap_denominator(group.event_probability)?;
    let mut draw = 0i64;
    loop {
        let gap = sample_geometric_gap_with_denominator(rng_state, gap_denominator) as i64;
        if gap >= total_draws - draw {
            return Ok(());
        }
        draw += gap;
        let shot = (draw / nsets as i64) as usize;
        let set_idx = (draw % nsets as i64) as usize;
        let row = group.event_rows[sample_categorical_row(rng_state, &group.event_probabilities)];
        let conditions = &group.conditions[set_idx];
        let assignment = &group.assignments[row];
        for (bit_idx, &condition) in conditions.iter().enumerate() {
            if packed_bit(assignment, bit_idx) {
                xor_packed_presampled_condition(value_words, shot_words, shot, condition);
            }
        }
        draw += 1;
    }
}

fn presample_rare_categorical_group_packed_sparse(
    scratch_hits: &mut Vec<(u32, u32)>,
    rng_state: &mut u64,
    group: &RareCategoricalSampleGroup,
    shots: usize,
) -> Result<()> {
    let nsets = group.conditions.len();
    if group.event_probability <= 0.0 || nsets == 0 {
        return Ok(());
    }
    let total_draws = (shots as i64) * (nsets as i64);
    let gap_denominator = geometric_gap_denominator(group.event_probability)?;
    let mut draw = 0i64;
    loop {
        let gap = sample_geometric_gap_with_denominator(rng_state, gap_denominator) as i64;
        if gap >= total_draws - draw {
            return Ok(());
        }
        draw += gap;
        let shot = (draw / nsets as i64) as u32;
        let set_idx = (draw % nsets as i64) as usize;
        let row = group.event_rows[sample_categorical_row(rng_state, &group.event_probabilities)];
        let conditions = &group.conditions[set_idx];
        let assignment = &group.assignments[row];
        for (bit_idx, &condition) in conditions.iter().enumerate() {
            if packed_bit(assignment, bit_idx) {
                scratch_hits.push((condition as u32 - 1, shot));
            }
        }
        draw += 1;
    }
}

fn presample_bernoulli_condition_packed(
    value_words: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    condition: i32,
    probability: f64,
    shots: usize,
) -> Result<()> {
    let p = check_probability(probability)?;
    if p <= 0.0 {
        return Ok(());
    }
    let base = packed_condition_offset(shot_words, condition, 0);
    generate_packed_biased_bits(
        &mut value_words[base..base + shot_words],
        shot_words,
        rng_state,
        p,
        shots,
    )
}

fn presample_low_probability_bernoulli_group_packed_dense(
    value_words: &mut [u64],
    shot_words: usize,
    rng_state: &mut u64,
    group: &BernoulliSampleGroup,
    shots: usize,
) -> Result<()> {
    let nconditions = group.conditions.len();
    if group.probability <= 0.0 || nconditions == 0 {
        return Ok(());
    }
    let total_draws = (shots as i64) * (nconditions as i64);
    let gap_denominator = geometric_gap_denominator(group.probability)?;
    let mut draw = 0i64;
    loop {
        let gap = sample_geometric_gap_with_denominator(rng_state, gap_denominator) as i64;
        if gap >= total_draws - draw {
            return Ok(());
        }
        draw += gap;
        let shot = (draw / nconditions as i64) as usize;
        let condition_idx = (draw % nconditions as i64) as usize;
        xor_packed_presampled_condition(
            value_words,
            shot_words,
            shot,
            group.conditions[condition_idx],
        );
        draw += 1;
    }
}

fn presample_low_probability_bernoulli_group_packed_sparse(
    scratch_hits: &mut Vec<(u32, u32)>,
    rng_state: &mut u64,
    group: &BernoulliSampleGroup,
    shots: usize,
) -> Result<()> {
    let nconditions = group.conditions.len();
    if group.probability <= 0.0 || nconditions == 0 {
        return Ok(());
    }
    let total_draws = (shots as i64) * (nconditions as i64);
    let gap_denominator = geometric_gap_denominator(group.probability)?;
    let mut draw = 0i64;
    loop {
        let gap = sample_geometric_gap_with_denominator(rng_state, gap_denominator) as i64;
        if gap >= total_draws - draw {
            return Ok(());
        }
        draw += gap;
        let shot = (draw / nconditions as i64) as u32;
        let condition_idx = (draw % nconditions as i64) as usize;
        scratch_hits.push((group.conditions[condition_idx] as u32 - 1, shot));
        draw += 1;
    }
}

// ==============================================================================
// Public entry points
// ==============================================================================

pub fn prepare_presampled_exogenous_packed(
    samples: &mut PackedPresampledExogenous,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    samples.nshots = 0;
    samples.nsymbols = program.nsymbols;
    samples.shot_words = 0;
    samples.exogenous_assigned_words = exogenous_assigned_words(program)?;
    samples.sparse_condition_template_words = sparse_condition_words(program)?;
    samples.sparse_condition_words = samples.sparse_condition_template_words.clone();
    samples.value_words.clear();
    samples.sparse_hit_offsets.clear();
    samples.sparse_hit_shots.clear();
    samples.next_rng_state = 0;
    Ok(())
}

pub fn resample_prepared_exogenous_packed_in_place(
    samples: &mut PackedPresampledExogenous,
    program: &FactoredInstructionProgram,
    shots: usize,
    seed: u64,
) -> Result<()> {
    if samples.nsymbols != program.nsymbols
        || samples.exogenous_assigned_words.len() != symbol_word_count(program.nsymbols)
        || samples.sparse_condition_template_words.len() != symbol_word_count(program.nsymbols)
    {
        return Err(TicitError::new(
            "packed presampled exogenous storage was not prepared for this program",
        ));
    }
    samples.nshots = shots;
    samples.shot_words = packed_shot_word_count(shots);
    // Not cleared: sparse rows are never materialized, residual rows are never
    // read, and each dense family below zeroes or fully rewrites its own rows.
    samples
        .value_words
        .resize(program.nsymbols * samples.shot_words, 0);
    samples.scratch_hits.clear();
    let dense_table = samples.nsymbols * samples.shot_words * 8 <= SPARSE_MIN_TABLE_BYTES;
    let shot_words = samples.shot_words;
    if dense_table {
        // Small bit-planes stay cache-resident, so the classic layout wins:
        // one bulk clear, every family scattered densely, no sparse set.
        samples.value_words.fill(0);
        samples.sparse_condition_words.fill(0);
        samples
            .sparse_condition_words
            .resize(samples.sparse_condition_template_words.len(), 0);
    } else {
        samples
            .sparse_condition_words
            .clone_from(&samples.sparse_condition_template_words);
    }

    // Chooses each skip-family group's sink before sampling, from its
    // expected draw count — deterministic, so it never depends on the drawn
    // hits. A dense group's rows are zeroed and scattered directly (leaving
    // the sparse set); a sparse group's hits go to the draw-order scratch.
    let expects_dense = |expected_draws: f64, nconditions: usize| {
        dense_table || expected_draws > (nconditions * (shot_words / 2).max(1)) as f64
    };

    let mut rng_state = seed;
    for distribution in &program.sampled_categorical_distributions {
        if !dense_table {
            for &condition in &distribution.conditions {
                zero_packed_row(&mut samples.value_words, samples.shot_words, condition);
            }
        }
        presample_categorical_distribution_packed(
            &mut samples.value_words,
            samples.shot_words,
            &mut rng_state,
            distribution,
            shots,
        );
    }
    for group in &program.sampled_rare_categorical_groups {
        let nsets = group.conditions.len();
        let nconditions: usize = group.conditions.iter().map(Vec::len).sum();
        let expected_draws = (shots * nsets) as f64 * group.event_probability;
        if expects_dense(expected_draws, nconditions) {
            if !dense_table {
                for conditions in &group.conditions {
                    for &condition in conditions {
                        zero_packed_row(&mut samples.value_words, samples.shot_words, condition);
                        samples.sparse_condition_words[symbol_word_index(condition)] &=
                            !symbol_bit_mask(condition);
                    }
                }
            }
            presample_rare_categorical_group_packed_dense(
                &mut samples.value_words,
                samples.shot_words,
                &mut rng_state,
                group,
                shots,
            )?;
        } else {
            presample_rare_categorical_group_packed_sparse(
                &mut samples.scratch_hits,
                &mut rng_state,
                group,
                shots,
            )?;
        }
    }
    for i in 0..program.sampled_bernoulli_conditions.len() {
        if !dense_table {
            zero_packed_row(
                &mut samples.value_words,
                samples.shot_words,
                program.sampled_bernoulli_conditions[i],
            );
        }
        presample_bernoulli_condition_packed(
            &mut samples.value_words,
            samples.shot_words,
            &mut rng_state,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i],
            shots,
        )?;
    }
    for group in &program.sampled_low_probability_bernoulli_groups {
        let nconditions = group.conditions.len();
        let expected_draws = (shots * nconditions) as f64 * group.probability;
        if expects_dense(expected_draws, nconditions) {
            if !dense_table {
                for &condition in &group.conditions {
                    zero_packed_row(&mut samples.value_words, samples.shot_words, condition);
                    samples.sparse_condition_words[symbol_word_index(condition)] &=
                        !symbol_bit_mask(condition);
                }
            }
            presample_low_probability_bernoulli_group_packed_dense(
                &mut samples.value_words,
                samples.shot_words,
                &mut rng_state,
                group,
                shots,
            )?;
        } else {
            presample_low_probability_bernoulli_group_packed_sparse(
                &mut samples.scratch_hits,
                &mut rng_state,
                group,
                shots,
            )?;
        }
    }
    samples.next_rng_state = rng_state;

    let mut scratch_cursors = std::mem::take(&mut samples.scratch_cursors);
    let mut sparse_hit_offsets = std::mem::take(&mut samples.sparse_hit_offsets);
    let mut sparse_hit_shots = std::mem::take(&mut samples.sparse_hit_shots);
    build_sparse_condition_hits(
        samples.nsymbols,
        &samples.scratch_hits,
        &mut scratch_cursors,
        &mut sparse_hit_offsets,
        &mut sparse_hit_shots,
    );
    samples.scratch_cursors = scratch_cursors;
    samples.sparse_hit_offsets = sparse_hit_offsets;
    samples.sparse_hit_shots = sparse_hit_shots;
    Ok(())
}

/// Below this table size the whole bit-plane stays cache-resident, so dense
/// row XORs beat per-bit hit application regardless of density.
const SPARSE_MIN_TABLE_BYTES: usize = 1 << 20;

/// Zeroes one dense row ahead of a family that XORs or ORs into it.
fn zero_packed_row(value_words: &mut [u64], shot_words: usize, condition: i32) {
    let base = (condition - 1) as usize * shot_words;
    value_words[base..base + shot_words].fill(0);
}

#[cfg(test)]
pub fn presample_exogenous_packed(
    program: &FactoredInstructionProgram,
    shots: usize,
    seed: u64,
) -> Result<PackedPresampledExogenous> {
    let mut samples = PackedPresampledExogenous::default();
    prepare_presampled_exogenous_packed(&mut samples, program)?;
    resample_prepared_exogenous_packed_in_place(&mut samples, program, shots, seed)?;
    Ok(samples)
}
