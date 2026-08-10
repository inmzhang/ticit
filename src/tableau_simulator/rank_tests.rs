//! Rank-behavior tests (design spec §4, item 4).

use crate::{Pauli, PauliString, SimError, TableauSimulator};

#[test]
fn diagonal_t_does_not_grow_rank() {
    // `T` about an axis that is diagonal in the frame (a = 0) merges both
    // branches into one label: rank stays 1.
    let mut sim = TableauSimulator::with_seed(1, 0);
    assert_eq!(sim.rank(), 1);
    sim.t(0).unwrap(); // Z on |0⟩ is frame-diagonal
    assert_eq!(sim.rank(), 1);
    sim.t(0).unwrap();
    assert_eq!(sim.rank(), 1);
}

#[test]
fn diagonal_t_leaves_a_multi_term_state_alone() {
    // The frame-diagonal `T` path multiplies each label by a unit-modulus
    // factor in place, so on a state that already carries several terms it must
    // change neither the label set nor any observable that ignores the rotated
    // qubit.
    let mut sim = TableauSimulator::with_seed(2, 0);
    sim.h(0);
    sim.t(0).unwrap(); // magic on qubit 0: rank 2, ⟨X_0⟩ = cos(π/4)
    let expectation_before = sim.peek_x(0).unwrap();

    sim.t(1).unwrap(); // Z_1 is frame-diagonal on the untouched qubit 1
    assert_eq!(sim.rank(), 2, "a diagonal T must not disturb the label set");
    assert!((sim.peek_x(0).unwrap() - expectation_before).abs() < 1e-12);
    assert!((sim.peek_z(1).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn ccz_agrees_between_the_direct_and_rollback_paths() {
    // `ccz` applies its seven rotations straight to the simulator when neither
    // failure mode is reachable, and falls back to a rollback clone otherwise.
    // A pruning threshold above the default (harmless at 1e-11, which is still
    // far below any real amplitude) selects the clone path; both must produce
    // the same state.
    let build = || {
        let mut sim = TableauSimulator::with_seed(3, 0);
        for q in 0..3 {
            sim.h(q);
        }
        sim
    };
    let mut direct = build();
    direct.ccz(0, 1, 2).unwrap();

    let mut rollback = build();
    rollback.set_prune_epsilon(1e-11);
    rollback.ccz(0, 1, 2).unwrap();

    assert_eq!(direct.rank(), rollback.rank());
    for (lhs, rhs) in direct.state_vector().iter().zip(rollback.state_vector()) {
        assert!((lhs - rhs).norm() < 1e-12, "{lhs} vs {rhs}");
    }
}

#[test]
fn off_diagonal_t_doubles_then_merges() {
    let mut sim = TableauSimulator::with_seed(1, 0);
    sim.h(0); // frame now maps Z → X, so T_Z is off-diagonal
    sim.t(0).unwrap();
    assert_eq!(sim.rank(), 2, "off-diagonal T should double the rank");
    // T then T† about the same axis is the identity and re-merges to rank 1.
    sim.t_dag(0).unwrap();
    assert_eq!(sim.rank(), 1, "T†T should collapse back to rank 1");
}

#[test]
fn generic_t_count_bounds_rank_by_two_to_the_t() {
    // A `T` on a fresh Hadamard-rotated qubit doubles the rank each time.
    let n = 6;
    let mut sim = TableauSimulator::with_seed(n, 0);
    for i in 0..n {
        sim.h(i);
        sim.t(i).unwrap();
        assert_eq!(
            sim.rank(),
            1 << (i + 1),
            "rank should be 2^(t) after {} Ts",
            i + 1
        );
    }
}

#[test]
fn measurement_can_reduce_rank() {
    // Magic injected on qubit 0, then measuring its Z collapses the branch.
    let mut sim = TableauSimulator::with_seed(1, 0);
    sim.h(0);
    sim.t(0).unwrap();
    assert_eq!(sim.rank(), 2);
    // Measuring Z_0 is frame-random here; either outcome projects to rank 1.
    sim.postselect_z(0, false).unwrap();
    assert_eq!(sim.rank(), 1);
}

#[test]
fn rank_overflow_is_reported() {
    let mut sim = TableauSimulator::with_seed(8, 0);
    sim.set_rank_cap(4);
    // Each off-diagonal T doubles the rank: 1 → 2 → 4 → 8 > cap.
    sim.h(0);
    sim.t(0).unwrap(); // rank 2
    sim.h(1);
    sim.t(1).unwrap(); // rank 4 (== cap, still ok)
    sim.h(2);
    let err = sim.t(2).unwrap_err(); // rank 8 > cap
    assert!(matches!(err, SimError::RankOverflow { rank: 8, cap: 4 }));
}

#[test]
fn pruning_cannot_install_an_empty_state() {
    let mut sim = TableauSimulator::with_seed(1, 0);
    sim.set_prune_epsilon(0.9);

    let error = sim
        .postselect_x(0, false)
        .expect_err("random-basis projection coefficients are below the threshold");

    assert_eq!(error, SimError::EmptyStateAfterPruning { epsilon: 0.9 });
    assert_eq!(sim.rank(), 1, "failed finalization must preserve the state");
}

#[test]
fn wide_registers_use_multiword_labels() {
    // n = 70 spans two u64 label words; the frame math must still work.
    let mut sim = TableauSimulator::with_seed(70, 0);
    sim.h(65);
    sim.t(65).unwrap();
    assert_eq!(sim.rank(), 2);
    // Z_65 is frame-random here; projecting collapses back to rank 1.
    sim.postselect_z(65, false).unwrap();
    assert_eq!(sim.rank(), 1);
}

#[test]
fn heap_allocated_labels_survive_the_full_pipeline() {
    // Labels keep eight words inline; n = 600 needs ten, so every label here is
    // heap-allocated. That is the branch of `Label`'s hand-written `Clone` the
    // other tests never reach.
    let n = 600;
    let mut sim = TableauSimulator::with_seed(n, 0);
    sim.h(575);
    sim.t(575).unwrap();
    assert_eq!(sim.rank(), 2);
    sim.h(577);
    sim.t(577).unwrap();
    assert_eq!(sim.rank(), 4);
    // Frame-random on both magic qubits: each projection halves the rank.
    sim.postselect_z(575, false).unwrap();
    sim.postselect_z(577, false).unwrap();
    assert_eq!(sim.rank(), 1);
    assert!((sim.peek_z(575).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn qubit_count_grows_on_demand_across_word_boundary() {
    // Start with one label word; an op on qubit 100 grows to two words.
    let mut sim = TableauSimulator::with_seed(2, 0);
    assert_eq!(sim.num_qubits(), 2);
    sim.cx(0, 100).expect("distinct CNOT operands");
    assert!(sim.num_qubits() >= 101);
    assert_eq!(
        sim.rank(),
        1,
        "a Clifford must not disturb the amplitude map"
    );
    // A T on the freshly added, entangled qubit still behaves.
    sim.h(100);
    sim.t(100).unwrap();
    assert_eq!(sim.rank(), 2);
}

#[test]
fn growth_across_every_label_width_boundary_preserves_the_state() {
    // Labels are held at one of four fixed widths, plus a heap fallback past
    // 512 qubits. Growing *within* a class is free — the surplus words of a
    // key were already zero — but each boundary below changes the key type and
    // rebuilds the map, and a re-keying bug there would drop or alias terms
    // rather than fail loudly. These four pairs are the complete set of
    // boundaries: 1→2 words, 2→4, 4→8, and 8→heap.
    for (start, grown) in [(64usize, 65usize), (128, 129), (256, 257), (512, 513)] {
        // `h; t` on two qubits: rank 4, and `T|+⟩` on each, whose `⟨X⟩` and
        // `⟨Y⟩` are `1/√2` — values that pin the actual amplitudes, not just
        // the label set.
        let mut sim = TableauSimulator::with_seed(start, 0);
        for q in 0..2 {
            sim.h(q);
            sim.t(q).unwrap();
        }
        assert_eq!(sim.rank(), 4, "start n = {start}");
        let readouts = |sim: &TableauSimulator| {
            [
                sim.peek_x(0).unwrap(),
                sim.peek_y(0).unwrap(),
                sim.peek_x(1).unwrap(),
                sim.peek_z(1).unwrap(),
            ]
        };
        let before = readouts(&sim);

        // The first qubit of the next word — the op that grows the register.
        sim.h(grown - 1);
        assert_eq!(sim.num_qubits(), grown, "grown n = {grown}");
        assert_eq!(sim.rank(), 4, "growth must not disturb the map");
        let after = readouts(&sim);
        for (b, a) in before.iter().zip(&after) {
            assert!(
                (b - a).abs() < 1e-12,
                "readout moved across the {start} → {grown} boundary: {b} vs {a}"
            );
        }

        // And the wider keys must still drive the hot paths: a frame-random
        // projection on each magic qubit halves the rank and leaves an
        // eigenstate behind.
        for q in 0..2 {
            sim.postselect_z(q, false).unwrap();
        }
        assert_eq!(sim.rank(), 1, "grown n = {grown}");
        assert!((sim.peek_z(0).unwrap() - 1.0).abs() < 1e-12);
    }
}

/// An observable wider than the register grows it rather than being rejected —
/// the counterpart to the `peek_*` reads, which cannot grow and so must reject.
#[test]
fn a_wide_observable_grows_the_register() {
    let mut sim = TableauSimulator::with_seed(1, 0);
    let wide = PauliString::single(5, 4, Pauli::Z);
    sim.measure_observable(&wide).unwrap();
    assert_eq!(sim.num_qubits(), 5);

    let mut sim = TableauSimulator::with_seed(1, 0);
    sim.t_pauli(&wide, false).unwrap();
    assert_eq!(sim.num_qubits(), 5);
}
