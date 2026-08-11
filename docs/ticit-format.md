# The `.tic` circuit format

`.tic` is a UTF-8, line-oriented quantum-circuit format. It uses familiar
Clifft/Stim-style gate mnemonics and record targets while defining
postselection directly in the source.

Blank lines are ignored. `#` starts a comment outside an instruction tag.
Targets are whitespace-separated; numeric targets are zero-based qubit
indices. Arguments appear in parentheses after the mnemonic.

```text
H 0
CX 0 1
X_ERROR(0.001) 0 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
```

`REPEAT count { ... }` repeats a block. Measurement records use relative
lookback targets such as `rec[-1]`. Pauli products use `*`, for example
`MPP X0*Z1`. Instruction tags such as `M[tag] 0` are accepted and ignored.

The instruction set covers Clifford gates and aliases, Pauli rotations and
products, measurements/resets, classical feedback, probabilistic Pauli and
depolarizing channels, heralded channels, detector/observable annotations,
expectation values, coordinate annotations, padding, and ticks. Unsupported
mnemonics and malformed targets are rejected with a 1-based source line.

## Postselection

`DETECTOR` declares the parity of its record targets. It does not discard a
shot by itself.

`DISCARD` has exactly the same syntax and parity semantics, but marks that
detector for postselection:

```text
M 0 1
DISCARD rec[-1] rec[-2]
```

The Rust API also accepts `SamplerOptions::postselection_mask`, a flat `Vec<u8>`
indexed in source order across both `DETECTOR` and `DISCARD` declarations. A
nonzero entry marks that detector for postselection. A short mask leaves later
detectors unmarked; extra entries are ignored.

The effective postselection set is the union of source `DISCARD` declarations
and nonzero mask entries. A shot is discarded when any detector in that union
has odd parity.
