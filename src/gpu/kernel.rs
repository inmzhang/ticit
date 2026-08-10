//! cuTile device kernels for the optional GPU backend.

// cuTile 0.2 does not lower compound assignments in device functions.
#![allow(clippy::assign_op_pattern, clippy::too_many_arguments)]

#[cutile::module]
mod kernels {
    use cutile::core::*;

    const DIM: i32 = 16;
    const META_WORDS: i32 = 12;
    const PARAM_WORDS: i32 = 4;
    const CONTROL_WORDS: i32 = 1;
    const DRAWS_PER_GROUP: i32 = 16;
    const SPARSE_META_WORDS: i32 = 7;
    const INV_SQRT2: f32 = 0.70710677;

    fn load_u64(pointer: *mut u64, offset: i32) -> u64 {
        let source: PointerTile<*mut u64, { [] }> = pointer_to_tile(pointer).offset(offset);
        let (value, _): (Tile<u64, { [] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None::<u64>,
            None,
            Latency::<0>,
        );
        tile_to_scalar(value)
    }

    fn store_u64(pointer: *mut u64, offset: i32, value: Tile<u64, { [] }>) {
        let destination: PointerTile<*mut u64, { [] }> = pointer_to_tile(pointer).offset(offset);
        store_ptr_tko(
            destination,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }

    fn load_i32(pointer: *mut i32, offset: i32) -> i32 {
        let source: PointerTile<*mut i32, { [] }> = pointer_to_tile(pointer).offset(offset);
        let (value, _): (Tile<i32, { [] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None::<i32>,
            None,
            Latency::<0>,
        );
        tile_to_scalar(value)
    }

    fn load_f32(pointer: *mut f32, offset: i32) -> f32 {
        let source: PointerTile<*mut f32, { [] }> = pointer_to_tile(pointer).offset(offset);
        let (value, _): (Tile<f32, { [] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None::<f32>,
            None,
            Latency::<0>,
        );
        tile_to_scalar(value)
    }

    fn load_randoms<const SHOTS: i32>(
        pointer: *mut f32,
        row: i32,
        stride: i32,
        shot_start: i32,
        lanes: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
    ) -> Tile<f32, { [SHOTS] }> {
        let source: PointerTile<*mut f32, { [] }> =
            pointer_to_tile(pointer).offset(row * stride + shot_start);
        let base: PointerTile<*mut f32, { [1] }> = source.reshape(const_shape![1]);
        let sources: PointerTile<*mut f32, { [SHOTS] }> = base.broadcast(const_shape![SHOTS]);
        let sources: PointerTile<*mut f32, { [SHOTS] }> = sources.offset_tile(lanes);
        let (values, _): (Tile<f32, { [SHOTS] }>, Token) = load_ptr_tko(
            sources,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        values
    }

    fn load_u64_row<const SHOTS: i32>(
        pointer: *mut u64,
        row: i32,
        stride: i32,
        shot_start: i32,
        lanes: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
    ) -> Tile<u64, { [SHOTS] }> {
        let source: PointerTile<*mut u64, { [] }> =
            pointer_to_tile(pointer).offset(row * stride + shot_start);
        let base: PointerTile<*mut u64, { [1] }> = source.reshape(const_shape![1]);
        let sources: PointerTile<*mut u64, { [SHOTS] }> = base.broadcast(const_shape![SHOTS]);
        let sources: PointerTile<*mut u64, { [SHOTS] }> = sources.offset_tile(lanes);
        let (values, _): (Tile<u64, { [SHOTS] }>, Token) = load_ptr_tko(
            sources,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            Some(0u64),
            None,
            Latency::<0>,
        );
        values
    }

    fn store_u64_row<const SHOTS: i32>(
        pointer: *mut u64,
        row: i32,
        stride: i32,
        shot_start: i32,
        lanes: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
        values: Tile<u64, { [SHOTS] }>,
    ) {
        let destination: PointerTile<*mut u64, { [] }> =
            pointer_to_tile(pointer).offset(row * stride + shot_start);
        let base: PointerTile<*mut u64, { [1] }> = destination.reshape(const_shape![1]);
        let destinations: PointerTile<*mut u64, { [SHOTS] }> = base.broadcast(const_shape![SHOTS]);
        let destinations: PointerTile<*mut u64, { [SHOTS] }> = destinations.offset_tile(lanes);
        store_ptr_tko(
            destinations,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            None,
            Latency::<0>,
        );
    }

    fn store_f32_row<const SHOTS: i32>(
        pointer: *mut f32,
        row: i32,
        stride: i32,
        shot_start: i32,
        lanes: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
        values: Tile<f32, { [SHOTS] }>,
    ) {
        let destination: PointerTile<*mut f32, { [] }> =
            pointer_to_tile(pointer).offset(row * stride + shot_start);
        let base: PointerTile<*mut f32, { [1] }> = destination.reshape(const_shape![1]);
        let destinations: PointerTile<*mut f32, { [SHOTS] }> = base.broadcast(const_shape![SHOTS]);
        let destinations: PointerTile<*mut f32, { [SHOTS] }> = destinations.offset_tile(lanes);
        store_ptr_tko(
            destinations,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            None,
            Latency::<0>,
        );
    }

    fn load_u64_offsets<const SHOTS: i32>(
        pointer: *mut u64,
        offsets: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
    ) -> Tile<u64, { [SHOTS] }> {
        let base: PointerTile<*mut u64, { [1] }> =
            pointer_to_tile(pointer).reshape(const_shape![1]);
        let sources: PointerTile<*mut u64, { [SHOTS] }> = base.broadcast(const_shape![SHOTS]);
        let sources: PointerTile<*mut u64, { [SHOTS] }> = sources.offset_tile(offsets);
        let (values, _): (Tile<u64, { [SHOTS] }>, Token) = load_ptr_tko(
            sources,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            Some(0u64),
            None,
            Latency::<0>,
        );
        values
    }

    fn mix_random(
        mut value: Tile<u64, { [64] }>,
        multiplier0: u64,
        multiplier1: u64,
    ) -> Tile<u64, { [64] }> {
        let shift30: Tile<u64, { [64] }> = broadcast_scalar(30u64, const_shape![64]);
        let shift27: Tile<u64, { [64] }> = broadcast_scalar(27u64, const_shape![64]);
        let shift31: Tile<u64, { [64] }> = broadcast_scalar(31u64, const_shape![64]);
        let multiplier0: Tile<u64, { [64] }> = broadcast_scalar(multiplier0, const_shape![64]);
        let multiplier1: Tile<u64, { [64] }> = broadcast_scalar(multiplier1, const_shape![64]);
        value = xori(value, shri(value, shift30)) * multiplier0;
        value = xori(value, shri(value, shift27)) * multiplier1;
        xori(value, shri(value, shift31))
    }

    fn sparse_bits(
        state: Tile<u64, { [64] }>,
        multiplier0: u64,
        multiplier1: u64,
    ) -> Tile<u64, { [64] }> {
        let shift: Tile<u64, { [64] }> = broadcast_scalar(40u64, const_shape![64]);
        shri(mix_random(state, multiplier0, multiplier1), shift)
    }

    fn sparse_uniform(
        state: Tile<u64, { [64] }>,
        multiplier0: u64,
        multiplier1: u64,
    ) -> Tile<f32, { [64] }> {
        let bits: Tile<f32, { [64] }> = convert_tile(sparse_bits(state, multiplier0, multiplier1));
        let half: Tile<f32, { [64] }> = broadcast_scalar(0.5f32, const_shape![64]);
        let scale: Tile<f32, { [64] }> =
            broadcast_scalar(1.0f32 / 16_777_216.0f32, const_shape![64]);
        (bits + half) * scale
    }

    fn parity_step(value: Tile<u64, { [64, 16] }>, shift: u64) -> Tile<u64, { [64, 16] }> {
        let shifts: Tile<u64, { [64, 16] }> = broadcast_scalar(shift, const_shape![64, DIM]);
        xori(value, shri(value, shifts))
    }

    fn parity(mut value: Tile<u64, { [64, 16] }>) -> Tile<u64, { [64, 16] }> {
        // Basis indices and Pauli masks are four bits wide in this kernel.
        value = parity_step(value, 2u64);
        value = parity_step(value, 1u64);
        let one: Tile<u64, { [64, 16] }> = constant(1u64, const_shape![64, DIM]);
        andi(value, one)
    }

    fn parity_shot_step<const SHOTS: i32>(
        value: Tile<u64, { [SHOTS] }>,
        shift: u64,
    ) -> Tile<u64, { [SHOTS] }> {
        let shifts: Tile<u64, { [SHOTS] }> = broadcast_scalar(shift, const_shape![SHOTS]);
        xori(value, shri(value, shifts))
    }

    fn parity_shots<const SHOTS: i32>(mut value: Tile<u64, { [SHOTS] }>) -> Tile<u64, { [SHOTS] }> {
        value = parity_shot_step(value, 32u64);
        value = parity_shot_step(value, 16u64);
        value = parity_shot_step(value, 8u64);
        value = parity_shot_step(value, 4u64);
        value = parity_shot_step(value, 2u64);
        value = parity_shot_step(value, 1u64);
        let one: Tile<u64, { [SHOTS] }> = constant(1u64, const_shape![SHOTS]);
        andi(value, one)
    }

    fn flip0(value: Tile<f32, { [64, 16] }>) -> Tile<f32, { [64, 16] }> {
        let shaped: Tile<f32, { [64, 8, 2, 1] }> = value.reshape(const_shape![64, 8, 2, 1]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [64, 8, 1, 1] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [64, 8, 1, 1] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [64, 8, 2, 1] }> = cat(high, low, 2);
        swapped.reshape(const_shape![64, DIM])
    }

    fn flip1(value: Tile<f32, { [64, 16] }>) -> Tile<f32, { [64, 16] }> {
        let shaped: Tile<f32, { [64, 4, 2, 2] }> = value.reshape(const_shape![64, 4, 2, 2]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [64, 4, 1, 2] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [64, 4, 1, 2] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [64, 4, 2, 2] }> = cat(high, low, 2);
        swapped.reshape(const_shape![64, DIM])
    }

    fn flip2(value: Tile<f32, { [64, 16] }>) -> Tile<f32, { [64, 16] }> {
        let shaped: Tile<f32, { [64, 2, 2, 4] }> = value.reshape(const_shape![64, 2, 2, 4]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [64, 2, 1, 4] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [64, 2, 1, 4] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [64, 2, 2, 4] }> = cat(high, low, 2);
        swapped.reshape(const_shape![64, DIM])
    }

    fn flip3(value: Tile<f32, { [64, 16] }>) -> Tile<f32, { [64, 16] }> {
        let shaped: Tile<f32, { [64, 1, 2, 8] }> = value.reshape(const_shape![64, 1, 2, 8]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [64, 1, 1, 8] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [64, 1, 1, 8] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [64, 1, 2, 8] }> = cat(high, low, 2);
        swapped.reshape(const_shape![64, DIM])
    }

    fn flip_mask(mut value: Tile<f32, { [64, 16] }>, mask: u64) -> Tile<f32, { [64, 16] }> {
        if mask & 1u64 != 0u64 {
            value = flip0(value);
        }
        if mask & 2u64 != 0u64 {
            value = flip1(value);
        }
        if mask & 4u64 != 0u64 {
            value = flip2(value);
        }
        if mask & 8u64 != 0u64 {
            value = flip3(value);
        }
        value
    }

    fn flip_dynamic(mut value: Tile<f32, { [64, 16] }>, bit: u64) -> Tile<f32, { [64, 16] }> {
        if bit == 0u64 {
            value = flip0(value);
        }
        if bit == 1u64 {
            value = flip1(value);
        }
        if bit == 2u64 {
            value = flip2(value);
        }
        if bit == 3u64 {
            value = flip3(value);
        }
        value
    }

    // One compact state per CTA avoids padding max_k <= 7 to 1,024 amplitudes.
    fn flip_compact<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 128] }>,
    ) -> Tile<f32, { [1, 128] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(high, low, 2);
        swapped.reshape(const_shape![1, 128])
    }

    fn flip_mask_compact(mut value: Tile<f32, { [1, 128] }>, mask: u64) -> Tile<f32, { [1, 128] }> {
        if mask & 1u64 != 0u64 {
            value = flip_compact::<64, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = flip_compact::<32, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = flip_compact::<16, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = flip_compact::<8, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = flip_compact::<4, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = flip_compact::<2, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = flip_compact::<1, 64>(value);
        }
        value
    }

    fn flip_dynamic_compact(
        mut value: Tile<f32, { [1, 128] }>,
        bit: u64,
    ) -> Tile<f32, { [1, 128] }> {
        if bit == 0u64 {
            value = flip_compact::<64, 1>(value);
        } else if bit == 1u64 {
            value = flip_compact::<32, 2>(value);
        } else if bit == 2u64 {
            value = flip_compact::<16, 4>(value);
        } else if bit == 3u64 {
            value = flip_compact::<8, 8>(value);
        } else if bit == 4u64 {
            value = flip_compact::<4, 16>(value);
        } else if bit == 5u64 {
            value = flip_compact::<2, 32>(value);
        } else if bit == 6u64 {
            value = flip_compact::<1, 64>(value);
        }
        value
    }

    fn hadamard_compact<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 128] }>,
    ) -> Tile<f32, { [1, 128] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let transformed: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(low + high, low - high, 2);
        transformed.reshape(const_shape![1, 128])
    }

    fn hadamard_mask_compact(
        mut value: Tile<f32, { [1, 128] }>,
        mask: u64,
    ) -> Tile<f32, { [1, 128] }> {
        if mask & 1u64 != 0u64 {
            value = hadamard_compact::<64, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = hadamard_compact::<32, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = hadamard_compact::<16, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = hadamard_compact::<8, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = hadamard_compact::<4, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = hadamard_compact::<2, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = hadamard_compact::<1, 64>(value);
        }
        value
    }

    fn parity_compact_step(value: Tile<u64, { [1, 128] }>, shift: u64) -> Tile<u64, { [1, 128] }> {
        let shifts: Tile<u64, { [1, 128] }> = broadcast_scalar(shift, const_shape![1, 128]);
        xori(value, shri(value, shifts))
    }

    fn parity_compact(mut value: Tile<u64, { [1, 128] }>) -> Tile<u64, { [1, 128] }> {
        // Basis indices and Pauli masks are seven bits wide in this kernel.
        value = parity_compact_step(value, 4u64);
        let nibble_mask: Tile<u64, { [1, 128] }> = constant(15u64, const_shape![1, 128]);
        let nibble = andi(value, nibble_mask);
        let table: Tile<u64, { [1, 128] }> = constant(0x6996u64, const_shape![1, 128]);
        let one: Tile<u64, { [1, 128] }> = constant(1u64, const_shape![1, 128]);
        andi(shri(table, nibble), one)
    }

    fn measurement_probability_compact(
        re: Tile<f32, { [1, 128] }>,
        im: Tile<f32, { [1, 128] }>,
        basis: Tile<u64, { [1, 128] }>,
        parameters: *mut f32,
        params: i32,
        xmask: u64,
        zmask: u64,
        pivot: u64,
        diagonal_phase_word: u64,
    ) -> Tile<f32, { [1] }> {
        let zero_state: Tile<f32, { [1, 128] }> = constant(0.0f32, const_shape![1, 128]);
        let one_state: Tile<f32, { [1, 128] }> = constant(1.0f32, const_shape![1, 128]);
        let negative_one_state: Tile<f32, { [1, 128] }> = constant(-1.0f32, const_shape![1, 128]);
        let inv_sqrt2_state: Tile<f32, { [1, 128] }> = constant(INV_SQRT2, const_shape![1, 128]);
        let zero_basis: Tile<u64, { [1, 128] }> = constant(0u64, const_shape![1, 128]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_true = if xmask == 0u64 {
            let odd = ne_tile(
                parity_compact(andi(basis, broadcast_scalar(zmask, const_shape![1, 128]))),
                zero_basis,
            );
            let target = diagonal_phase_word == 0u64;
            let selected = eq_tile(odd, broadcast_scalar(target, const_shape![1, 128]));
            reduce_sum(select(selected, re * re + im * im, zero_state), 1i32)
        } else {
            let partner_re = flip_mask_compact(re, xmask);
            let partner_im = flip_mask_compact(im, xmask);
            let odd = ne_tile(
                parity_compact(andi(basis, broadcast_scalar(zmask, const_shape![1, 128]))),
                zero_basis,
            );
            let direction = select(odd, one_state, negative_one_state);
            let cr =
                direction * broadcast_scalar(load_f32(parameters, params), const_shape![1, 128]);
            let ci = direction
                * broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 128]);
            let ar = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
            let ai = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
            let pivot_clear = eq_tile(
                andi(
                    basis,
                    broadcast_scalar(bit_mask(pivot), const_shape![1, 128]),
                ),
                zero_basis,
            );
            reduce_sum(select(pivot_clear, ar * ar + ai * ai, zero_state), 1i32)
        };
        minf(
            maxf(
                probability_true,
                zero_probability,
                nan::Enabled,
                ftz::Disabled,
            ),
            one_probability,
            nan::Enabled,
            ftz::Disabled,
        )
    }

    // ponytail: cuTile 0.2 cannot express generic-const shape arithmetic, so
    // the medium-state permutation shape is concrete. Add another shape only
    // when a representative circuit needs a larger resident state.
    fn flip_medium<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 1024] }>,
    ) -> Tile<f32, { [1, 1024] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(high, low, 2);
        swapped.reshape(const_shape![1, 1024])
    }

    fn flip_mask_medium(
        mut value: Tile<f32, { [1, 1024] }>,
        mask: u64,
    ) -> Tile<f32, { [1, 1024] }> {
        if mask & 1u64 != 0u64 {
            value = flip_medium::<512, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = flip_medium::<256, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = flip_medium::<128, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = flip_medium::<64, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = flip_medium::<32, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = flip_medium::<16, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = flip_medium::<8, 64>(value);
        }
        if mask & 128u64 != 0u64 {
            value = flip_medium::<4, 128>(value);
        }
        if mask & 256u64 != 0u64 {
            value = flip_medium::<2, 256>(value);
        }
        if mask & 512u64 != 0u64 {
            value = flip_medium::<1, 512>(value);
        }
        value
    }

    fn flip_dynamic_medium(
        mut value: Tile<f32, { [1, 1024] }>,
        bit: u64,
    ) -> Tile<f32, { [1, 1024] }> {
        if bit == 0u64 {
            value = flip_medium::<512, 1>(value);
        } else if bit == 1u64 {
            value = flip_medium::<256, 2>(value);
        } else if bit == 2u64 {
            value = flip_medium::<128, 4>(value);
        } else if bit == 3u64 {
            value = flip_medium::<64, 8>(value);
        } else if bit == 4u64 {
            value = flip_medium::<32, 16>(value);
        } else if bit == 5u64 {
            value = flip_medium::<16, 32>(value);
        } else if bit == 6u64 {
            value = flip_medium::<8, 64>(value);
        } else if bit == 7u64 {
            value = flip_medium::<4, 128>(value);
        } else if bit == 8u64 {
            value = flip_medium::<2, 256>(value);
        } else if bit == 9u64 {
            value = flip_medium::<1, 512>(value);
        }
        value
    }

    fn hadamard_medium<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 1024] }>,
    ) -> Tile<f32, { [1, 1024] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let transformed: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(low + high, low - high, 2);
        transformed.reshape(const_shape![1, 1024])
    }

    fn hadamard_mask_medium(
        mut value: Tile<f32, { [1, 1024] }>,
        mask: u64,
    ) -> Tile<f32, { [1, 1024] }> {
        if mask & 1u64 != 0u64 {
            value = hadamard_medium::<512, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = hadamard_medium::<256, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = hadamard_medium::<128, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = hadamard_medium::<64, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = hadamard_medium::<32, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = hadamard_medium::<16, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = hadamard_medium::<8, 64>(value);
        }
        if mask & 128u64 != 0u64 {
            value = hadamard_medium::<4, 128>(value);
        }
        if mask & 256u64 != 0u64 {
            value = hadamard_medium::<2, 256>(value);
        }
        if mask & 512u64 != 0u64 {
            value = hadamard_medium::<1, 512>(value);
        }
        value
    }

    fn parity_medium_step(value: Tile<u64, { [1, 1024] }>, shift: u64) -> Tile<u64, { [1, 1024] }> {
        let shifts: Tile<u64, { [1, 1024] }> = broadcast_scalar(shift, const_shape![1, 1024]);
        xori(value, shri(value, shifts))
    }

    fn parity_medium(mut value: Tile<u64, { [1, 1024] }>) -> Tile<u64, { [1, 1024] }> {
        // Basis indices and Pauli masks are ten bits wide in this kernel.
        value = parity_medium_step(value, 8u64);
        value = parity_medium_step(value, 4u64);
        let nibble_mask: Tile<u64, { [1, 1024] }> = constant(15u64, const_shape![1, 1024]);
        let nibble = andi(value, nibble_mask);
        let table: Tile<u64, { [1, 1024] }> = constant(0x6996u64, const_shape![1, 1024]);
        let one: Tile<u64, { [1, 1024] }> = constant(1u64, const_shape![1, 1024]);
        andi(shri(table, nibble), one)
    }

    fn measurement_probability_medium(
        re: Tile<f32, { [1, 1024] }>,
        im: Tile<f32, { [1, 1024] }>,
        basis: Tile<u64, { [1, 1024] }>,
        parameters: *mut f32,
        params: i32,
        xmask: u64,
        zmask: u64,
        pivot: u64,
        diagonal_phase_word: u64,
    ) -> Tile<f32, { [1] }> {
        let zero_state: Tile<f32, { [1, 1024] }> = constant(0.0f32, const_shape![1, 1024]);
        let one_state: Tile<f32, { [1, 1024] }> = constant(1.0f32, const_shape![1, 1024]);
        let negative_one_state: Tile<f32, { [1, 1024] }> = constant(-1.0f32, const_shape![1, 1024]);
        let inv_sqrt2_state: Tile<f32, { [1, 1024] }> = constant(INV_SQRT2, const_shape![1, 1024]);
        let zero_basis: Tile<u64, { [1, 1024] }> = constant(0u64, const_shape![1, 1024]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_true = if xmask == 0u64 {
            let odd = ne_tile(
                parity_medium(andi(basis, broadcast_scalar(zmask, const_shape![1, 1024]))),
                zero_basis,
            );
            let target = diagonal_phase_word == 0u64;
            let selected = eq_tile(odd, broadcast_scalar(target, const_shape![1, 1024]));
            reduce_sum(select(selected, re * re + im * im, zero_state), 1i32)
        } else {
            let partner_re = flip_mask_medium(re, xmask);
            let partner_im = flip_mask_medium(im, xmask);
            let odd = ne_tile(
                parity_medium(andi(basis, broadcast_scalar(zmask, const_shape![1, 1024]))),
                zero_basis,
            );
            let direction = select(odd, one_state, negative_one_state);
            let cr =
                direction * broadcast_scalar(load_f32(parameters, params), const_shape![1, 1024]);
            let ci = direction
                * broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 1024]);
            let ar = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
            let ai = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
            let pivot_clear = eq_tile(
                andi(
                    basis,
                    broadcast_scalar(bit_mask(pivot), const_shape![1, 1024]),
                ),
                zero_basis,
            );
            reduce_sum(select(pivot_clear, ar * ar + ai * ai, zero_state), 1i32)
        };
        minf(
            maxf(
                probability_true,
                zero_probability,
                nan::Enabled,
                ftz::Disabled,
            ),
            one_probability,
            nan::Enabled,
            ftz::Disabled,
        )
    }

    fn measurement_probability_large(
        re: Tile<f32, { [1, 4096] }>,
        im: Tile<f32, { [1, 4096] }>,
        basis: Tile<u64, { [1, 4096] }>,
        parameters: *mut f32,
        params: i32,
        xmask: u64,
        zmask: u64,
        pivot: u64,
        diagonal_phase_word: u64,
    ) -> Tile<f32, { [1] }> {
        let zero_state: Tile<f32, { [1, 4096] }> = constant(0.0f32, const_shape![1, 4096]);
        let one_state: Tile<f32, { [1, 4096] }> = constant(1.0f32, const_shape![1, 4096]);
        let negative_one_state: Tile<f32, { [1, 4096] }> = constant(-1.0f32, const_shape![1, 4096]);
        let inv_sqrt2_state: Tile<f32, { [1, 4096] }> = constant(INV_SQRT2, const_shape![1, 4096]);
        let zero_basis: Tile<u64, { [1, 4096] }> = constant(0u64, const_shape![1, 4096]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_true = if xmask == 0u64 {
            let odd = ne_tile(
                parity_large(andi(basis, broadcast_scalar(zmask, const_shape![1, 4096]))),
                zero_basis,
            );
            let target = diagonal_phase_word == 0u64;
            let selected = eq_tile(odd, broadcast_scalar(target, const_shape![1, 4096]));
            reduce_sum(select(selected, re * re + im * im, zero_state), 1i32)
        } else {
            let partner_re = flip_mask_large(re, xmask);
            let partner_im = flip_mask_large(im, xmask);
            let odd = ne_tile(
                parity_large(andi(basis, broadcast_scalar(zmask, const_shape![1, 4096]))),
                zero_basis,
            );
            let direction = select(odd, one_state, negative_one_state);
            let cr =
                direction * broadcast_scalar(load_f32(parameters, params), const_shape![1, 4096]);
            let ci = direction
                * broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 4096]);
            let ar = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
            let ai = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
            let pivot_clear = eq_tile(
                andi(
                    basis,
                    broadcast_scalar(bit_mask(pivot), const_shape![1, 4096]),
                ),
                zero_basis,
            );
            reduce_sum(select(pivot_clear, ar * ar + ai * ai, zero_state), 1i32)
        };
        minf(
            maxf(
                probability_true,
                zero_probability,
                nan::Enabled,
                ftz::Disabled,
            ),
            one_probability,
            nan::Enabled,
            ftz::Disabled,
        )
    }

    fn flip_large<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 4096] }>,
    ) -> Tile<f32, { [1, 4096] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let swapped: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(high, low, 2);
        swapped.reshape(const_shape![1, 4096])
    }

    fn flip_mask_large(mut value: Tile<f32, { [1, 4096] }>, mask: u64) -> Tile<f32, { [1, 4096] }> {
        if mask & 1u64 != 0u64 {
            value = flip_large::<2048, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = flip_large::<1024, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = flip_large::<512, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = flip_large::<256, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = flip_large::<128, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = flip_large::<64, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = flip_large::<32, 64>(value);
        }
        if mask & 128u64 != 0u64 {
            value = flip_large::<16, 128>(value);
        }
        if mask & 256u64 != 0u64 {
            value = flip_large::<8, 256>(value);
        }
        if mask & 512u64 != 0u64 {
            value = flip_large::<4, 512>(value);
        }
        if mask & 1024u64 != 0u64 {
            value = flip_large::<2, 1024>(value);
        }
        if mask & 2048u64 != 0u64 {
            value = flip_large::<1, 2048>(value);
        }
        value
    }

    fn flip_dynamic_large(
        mut value: Tile<f32, { [1, 4096] }>,
        bit: u64,
    ) -> Tile<f32, { [1, 4096] }> {
        if bit == 0u64 {
            value = flip_large::<2048, 1>(value);
        } else if bit == 1u64 {
            value = flip_large::<1024, 2>(value);
        } else if bit == 2u64 {
            value = flip_large::<512, 4>(value);
        } else if bit == 3u64 {
            value = flip_large::<256, 8>(value);
        } else if bit == 4u64 {
            value = flip_large::<128, 16>(value);
        } else if bit == 5u64 {
            value = flip_large::<64, 32>(value);
        } else if bit == 6u64 {
            value = flip_large::<32, 64>(value);
        } else if bit == 7u64 {
            value = flip_large::<16, 128>(value);
        } else if bit == 8u64 {
            value = flip_large::<8, 256>(value);
        } else if bit == 9u64 {
            value = flip_large::<4, 512>(value);
        } else if bit == 10u64 {
            value = flip_large::<2, 1024>(value);
        } else if bit == 11u64 {
            value = flip_large::<1, 2048>(value);
        }
        value
    }

    fn hadamard_large<const OUTER: i32, const INNER: i32>(
        value: Tile<f32, { [1, 4096] }>,
    ) -> Tile<f32, { [1, 4096] }> {
        let shaped: Tile<f32, { [1, OUTER, 2, INNER] }> =
            value.reshape(const_shape![1, OUTER, 2, INNER]);
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let low: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, zero, zero]);
        let high: Tile<f32, { [1, OUTER, 1, INNER] }> = extract(shaped, [zero, zero, one, zero]);
        let transformed: Tile<f32, { [1, OUTER, 2, INNER] }> = cat(low + high, low - high, 2);
        transformed.reshape(const_shape![1, 4096])
    }

    fn hadamard_mask_large(
        mut value: Tile<f32, { [1, 4096] }>,
        mask: u64,
    ) -> Tile<f32, { [1, 4096] }> {
        if mask & 1u64 != 0u64 {
            value = hadamard_large::<2048, 1>(value);
        }
        if mask & 2u64 != 0u64 {
            value = hadamard_large::<1024, 2>(value);
        }
        if mask & 4u64 != 0u64 {
            value = hadamard_large::<512, 4>(value);
        }
        if mask & 8u64 != 0u64 {
            value = hadamard_large::<256, 8>(value);
        }
        if mask & 16u64 != 0u64 {
            value = hadamard_large::<128, 16>(value);
        }
        if mask & 32u64 != 0u64 {
            value = hadamard_large::<64, 32>(value);
        }
        if mask & 64u64 != 0u64 {
            value = hadamard_large::<32, 64>(value);
        }
        if mask & 128u64 != 0u64 {
            value = hadamard_large::<16, 128>(value);
        }
        if mask & 256u64 != 0u64 {
            value = hadamard_large::<8, 256>(value);
        }
        if mask & 512u64 != 0u64 {
            value = hadamard_large::<4, 512>(value);
        }
        if mask & 1024u64 != 0u64 {
            value = hadamard_large::<2, 1024>(value);
        }
        if mask & 2048u64 != 0u64 {
            value = hadamard_large::<1, 2048>(value);
        }
        value
    }

    fn parity_large_step(value: Tile<u64, { [1, 4096] }>, shift: u64) -> Tile<u64, { [1, 4096] }> {
        let shifts: Tile<u64, { [1, 4096] }> = broadcast_scalar(shift, const_shape![1, 4096]);
        xori(value, shri(value, shifts))
    }

    fn parity_large(mut value: Tile<u64, { [1, 4096] }>) -> Tile<u64, { [1, 4096] }> {
        value = parity_large_step(value, 8u64);
        value = parity_large_step(value, 4u64);
        let nibble_mask: Tile<u64, { [1, 4096] }> = constant(15u64, const_shape![1, 4096]);
        let nibble = andi(value, nibble_mask);
        let table: Tile<u64, { [1, 4096] }> = constant(0x6996u64, const_shape![1, 4096]);
        let one: Tile<u64, { [1, 4096] }> = constant(1u64, const_shape![1, 4096]);
        andi(shri(table, nibble), one)
    }

    fn bit_mask(bit: u64) -> u64 {
        let mut mask = 1u64;
        if bit == 1u64 {
            mask = 2u64;
        } else if bit == 2u64 {
            mask = 4u64;
        } else if bit == 3u64 {
            mask = 8u64;
        } else if bit == 4u64 {
            mask = 16u64;
        } else if bit == 5u64 {
            mask = 32u64;
        } else if bit == 6u64 {
            mask = 64u64;
        } else if bit == 7u64 {
            mask = 128u64;
        } else if bit == 8u64 {
            mask = 256u64;
        } else if bit == 9u64 {
            mask = 512u64;
        } else if bit == 10u64 {
            mask = 1024u64;
        } else if bit == 11u64 {
            mask = 2048u64;
        }
        mask
    }

    fn expression<const SHOTS: i32>(
        values: Tile<u64, { [SHOTS] }>,
        block_mask: u64,
        mask0: u64,
        mask1: u64,
        mask2: u64,
        mask3: u64,
        branches0: Tile<u64, { [SHOTS] }>,
        branches1: Tile<u64, { [SHOTS] }>,
        branches2: Tile<u64, { [SHOTS] }>,
        branches3: Tile<u64, { [SHOTS] }>,
    ) -> Tile<bool, { [SHOTS] }> {
        let selected = andi(values, broadcast_scalar(block_mask, const_shape![SHOTS]));
        let zero: Tile<u64, { [SHOTS] }> = constant(0u64, const_shape![SHOTS]);
        let base = ne_tile(selected, zero);
        let residual_bits = parity_shots(
            andi(branches0, broadcast_scalar(mask0, const_shape![SHOTS]))
                ^ andi(branches1, broadcast_scalar(mask1, const_shape![SHOTS]))
                ^ andi(branches2, broadcast_scalar(mask2, const_shape![SHOTS]))
                ^ andi(branches3, broadcast_scalar(mask3, const_shape![SHOTS])),
        );
        let residual = ne_tile(residual_bits, zero);
        ne_tile(base, residual)
    }

    fn instruction_expression<const SHOTS: i32>(
        metadata: *mut u64,
        values: Tile<u64, { [SHOTS] }>,
        instruction: i32,
        branches0: Tile<u64, { [SHOTS] }>,
        branches1: Tile<u64, { [SHOTS] }>,
        branches2: Tile<u64, { [SHOTS] }>,
        branches3: Tile<u64, { [SHOTS] }>,
    ) -> Tile<bool, { [SHOTS] }> {
        let meta = instruction * META_WORDS;
        expression(
            values,
            load_u64(metadata, meta + 11i32),
            load_u64(metadata, meta + 1i32),
            load_u64(metadata, meta + 2i32),
            load_u64(metadata, meta + 3i32),
            load_u64(metadata, meta + 4i32),
            branches0,
            branches1,
            branches2,
            branches3,
        )
    }

    fn instruction_expression_rows<const SHOTS: i32>(
        metadata: *mut u64,
        controls: *mut i32,
        expression_values: *mut u64,
        stride: i32,
        shot_start: i32,
        lanes: Tile<i32, { [SHOTS] }>,
        live: Tile<bool, { [SHOTS] }>,
        instruction: i32,
        branches0: Tile<u64, { [SHOTS] }>,
        branches1: Tile<u64, { [SHOTS] }>,
        branches2: Tile<u64, { [SHOTS] }>,
        branches3: Tile<u64, { [SHOTS] }>,
    ) -> Tile<bool, { [SHOTS] }> {
        let meta = instruction * META_WORDS;
        let values = load_u64_row(
            expression_values,
            load_i32(controls, instruction),
            stride,
            shot_start,
            lanes,
            live,
        );
        expression(
            values,
            load_u64(metadata, meta + 11i32),
            load_u64(metadata, meta + 1i32),
            load_u64(metadata, meta + 2i32),
            load_u64(metadata, meta + 3i32),
            load_u64(metadata, meta + 4i32),
            branches0,
            branches1,
            branches2,
            branches3,
        )
    }

    const WIDE_TILE: i32 = 1024;

    fn wide_dimension(mut k: i32) -> i32 {
        let mut dimension = 1i32;
        while k > 0i32 {
            dimension = dimension * 2i32;
            k = k - 1i32;
        }
        dimension
    }

    fn wide_bit_mask(bit: u64) -> u64 {
        let one: Tile<u64, { [] }> = scalar_to_tile(1u64);
        let shift: Tile<u64, { [] }> = scalar_to_tile(bit);
        tile_to_scalar(shli(one, shift, overflow::NoUnsignedWrap))
    }

    fn wide_insert_zero_bit(packed: Tile<u64, { [1024] }>, bit: u64) -> Tile<u64, { [1024] }> {
        let selector = wide_bit_mask(bit);
        let lower_mask = selector - 1u64;
        let lower = andi(
            packed,
            broadcast_scalar(lower_mask, const_shape![WIDE_TILE]),
        );
        let upper = packed - lower;
        ori(
            lower,
            shli(
                upper,
                broadcast_scalar(1u64, const_shape![WIDE_TILE]),
                overflow::NoUnsignedWrap,
            ),
        )
    }

    fn wide_parity_step(value: Tile<u64, { [1024] }>, shift: u64) -> Tile<u64, { [1024] }> {
        xori(
            value,
            shri(value, broadcast_scalar(shift, const_shape![WIDE_TILE])),
        )
    }

    fn wide_parity(mut value: Tile<u64, { [1024] }>) -> Tile<u64, { [1024] }> {
        value = wide_parity_step(value, 16u64);
        value = wide_parity_step(value, 8u64);
        value = wide_parity_step(value, 4u64);
        value = wide_parity_step(value, 2u64);
        value = wide_parity_step(value, 1u64);
        let one: Tile<u64, { [1024] }> = constant(1u64, const_shape![WIDE_TILE]);
        andi(value, one)
    }

    fn load_wide_f32(
        pointer: *mut f32,
        offsets: Tile<i32, { [1024] }>,
        live: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let base: PointerTile<*mut f32, { [1] }> =
            pointer_to_tile(pointer).reshape(const_shape![1]);
        let sources: PointerTile<*mut f32, { [1024] }> = base.broadcast(const_shape![WIDE_TILE]);
        let sources: PointerTile<*mut f32, { [1024] }> = sources.offset_tile(offsets);
        let (values, _): (Tile<f32, { [1024] }>, Token) = load_ptr_tko(
            sources,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        values
    }

    fn store_wide_f32(
        pointer: *mut f32,
        offsets: Tile<i32, { [1024] }>,
        live: Tile<bool, { [1024] }>,
        values: Tile<f32, { [1024] }>,
    ) {
        let base: PointerTile<*mut f32, { [1] }> =
            pointer_to_tile(pointer).reshape(const_shape![1]);
        let destinations: PointerTile<*mut f32, { [1024] }> =
            base.broadcast(const_shape![WIDE_TILE]);
        let destinations: PointerTile<*mut f32, { [1024] }> = destinations.offset_tile(offsets);
        store_ptr_tko(
            destinations,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            None,
            Latency::<0>,
        );
    }

    fn load_wide_f32_u64(
        pointer: *mut f32,
        base_offset: i32,
        offsets: Tile<u64, { [1024] }>,
        live: Tile<bool, { [1024] }>,
    ) -> Tile<f32, { [1024] }> {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(pointer).offset(base_offset);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let sources: PointerTile<*mut f32, { [1024] }> = base.broadcast(const_shape![WIDE_TILE]);
        let sources: PointerTile<*mut f32, { [1024] }> = sources.offset_tile(offsets);
        let (values, _): (Tile<f32, { [1024] }>, Token) = load_ptr_tko(
            sources,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        values
    }

    fn store_wide_f32_u64(
        pointer: *mut f32,
        base_offset: i32,
        offsets: Tile<u64, { [1024] }>,
        live: Tile<bool, { [1024] }>,
        values: Tile<f32, { [1024] }>,
    ) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(pointer).offset(base_offset);
        let base: PointerTile<*mut f32, { [1] }> = base.reshape(const_shape![1]);
        let destinations: PointerTile<*mut f32, { [1024] }> =
            base.broadcast(const_shape![WIDE_TILE]);
        let destinations: PointerTile<*mut f32, { [1024] }> = destinations.offset_tile(offsets);
        store_ptr_tko(
            destinations,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(live),
            None,
            Latency::<0>,
        );
    }

    fn store_wide_branch(
        branches: *mut u64,
        shot: i32,
        shots: i32,
        slot: u64,
        bit: u64,
        value: Tile<bool, { [1] }>,
        lanes: Tile<i32, { [1] }>,
        live: Tile<bool, { [1] }>,
    ) {
        let zero: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let set = select(value, broadcast_scalar(bit, const_shape![1]), zero);
        let word = if slot < 64u64 {
            0i32
        } else if slot < 128u64 {
            1i32
        } else if slot < 192u64 {
            2i32
        } else {
            3i32
        };
        let old = load_u64_row(branches, word, shots, shot, lanes, live);
        store_u64_row(branches, word, shots, shot, lanes, live, ori(old, set));
    }

    #[cutile::entry()]
    unsafe fn evaluate_exogenous_partials(
        partials: *mut u64,
        randoms: *mut f32,
        draw_transition_offsets: *mut i32,
        draw_base_masks: *mut u64,
        transition_upper: *mut f32,
        transition_masks: *mut u64,
        draw_count: i32,
        mask_words: i32,
        stride: i32,
        shots: i32,
    ) {
        let pid = get_tile_block_id();
        let shot_start = pid.0 * 64i32;
        let group = pid.1;
        let word = pid.2;
        let group_count = get_num_tile_blocks().1;
        let lanes: Tile<i32, { [64] }> = iota(const_shape![64]);
        let shot_indices: Tile<i32, { [64] }> =
            lanes + broadcast_scalar(shot_start, const_shape![64]);
        let live: Tile<bool, { [64] }> =
            lt_tile(shot_indices, broadcast_scalar(shots, const_shape![64]));
        let zero: Tile<u64, { [64] }> = constant(0u64, const_shape![64]);
        let mut value: Tile<u64, { [64] }> = zero;

        let draw_start = group * DRAWS_PER_GROUP;
        for local in 0i32..DRAWS_PER_GROUP {
            let draw = draw_start + local;
            if draw < draw_count {
                let base_mask: Tile<u64, { [64] }> = broadcast_scalar(
                    load_u64(draw_base_masks, draw * mask_words + word),
                    const_shape![64],
                );
                value = xori(value, base_mask);
                let uniform = load_randoms(randoms, draw, stride, shot_start, lanes, live);
                let start = load_i32(draw_transition_offsets, draw);
                let end = load_i32(draw_transition_offsets, draw + 1i32);
                for offset in 0i32..end - start {
                    let transition = start + offset;
                    let upper: Tile<f32, { [64] }> =
                        broadcast_scalar(load_f32(transition_upper, transition), const_shape![64]);
                    let hit: Tile<bool, { [64] }> = le_tile(uniform, upper);
                    let mask: Tile<u64, { [64] }> = broadcast_scalar(
                        load_u64(transition_masks, transition * mask_words + word),
                        const_shape![64],
                    );
                    value = xori(value, select(hit, mask, zero));
                }
            }
        }
        store_u64_row(
            partials,
            word * group_count + group,
            stride,
            shot_start,
            lanes,
            live,
            value,
        );
    }

    #[cutile::entry()]
    unsafe fn reduce_exogenous_partials(
        values: *mut u64,
        partials: *mut u64,
        constant_masks: *mut u64,
        group_count: i32,
        stride: i32,
        shots: i32,
    ) {
        let pid = get_tile_block_id();
        let shot_start = pid.0 * 64i32;
        let word = pid.1;
        let lanes: Tile<i32, { [64] }> = iota(const_shape![64]);
        let shot_indices: Tile<i32, { [64] }> =
            lanes + broadcast_scalar(shot_start, const_shape![64]);
        let live: Tile<bool, { [64] }> =
            lt_tile(shot_indices, broadcast_scalar(shots, const_shape![64]));
        let mut value: Tile<u64, { [64] }> =
            broadcast_scalar(load_u64(constant_masks, word), const_shape![64]);
        for group in 0i32..group_count {
            value = xori(
                value,
                load_u64_row(
                    partials,
                    word * group_count + group,
                    stride,
                    shot_start,
                    lanes,
                    live,
                ),
            );
        }
        store_u64_row(values, word, stride, shot_start, lanes, live, value);
    }

    #[cutile::entry()]
    unsafe fn reduce_block_counts(partials: *mut u64, counts: *mut u64, block_count: i32) {
        let pid = get_tile_block_id().0;
        let lanes: Tile<i32, { [256] }> = iota(const_shape![256]);
        let blocks = lanes + broadcast_scalar(pid * 256i32, const_shape![256]);
        let live = lt_tile(blocks, broadcast_scalar(block_count, const_shape![256]));
        let offsets = blocks + blocks;
        let discarded: Tile<u64, { [] }> =
            reduce_sum(load_u64_offsets(counts, offsets, live), 0i32);
        let logical: Tile<u64, { [] }> = reduce_sum(
            load_u64_offsets(counts, offsets + constant(1i32, const_shape![256]), live),
            0i32,
        );
        store_u64(partials, pid * 2i32, discarded);
        store_u64(partials, pid * 2i32 + 1i32, logical);
    }

    #[cutile::entry()]
    unsafe fn apply_sparse_exogenous(
        values: *mut u64,
        shot_block_offsets: *mut u64,
        group_metadata: *mut i32,
        group_keys: *mut u64,
        gap_thresholds: *mut u64,
        transition_upper: *mut f32,
        base_masks: *mut u64,
        transition_masks: *mut u64,
        seed: u64,
        shot_offset: u64,
        rng_gamma: u64,
        rng_multiplier0: u64,
        rng_multiplier1: u64,
        group_count: i32,
        mask_words: i32,
        stride: i32,
        shots: i32,
    ) {
        let pid = get_tile_block_id();
        let shot_start = pid.0 * 64i32;
        let word = pid.1;
        let lanes: Tile<i32, { [64] }> = iota(const_shape![64]);
        let shot_indices: Tile<i32, { [64] }> =
            lanes + broadcast_scalar(shot_start, const_shape![64]);
        let live: Tile<bool, { [64] }> =
            lt_tile(shot_indices, broadcast_scalar(shots, const_shape![64]));
        let shot_lanes: Tile<u64, { [64] }> = iota(const_shape![64]);
        let shot_global = shot_lanes
            + broadcast_scalar(
                load_u64(shot_block_offsets, pid.0) + shot_offset,
                const_shape![64],
            );
        let zero_i32: Tile<i32, { [64] }> = constant(0i32, const_shape![64]);
        let one_i32: Tile<i32, { [64] }> = constant(1i32, const_shape![64]);
        let zero_u64: Tile<u64, { [64] }> = constant(0u64, const_shape![64]);
        let one_u64: Tile<u64, { [64] }> = constant(1u64, const_shape![64]);
        let mut value = load_u64_row(values, word, stride, shot_start, lanes, live);

        for group in 0i32..group_count {
            let metadata = group * SPARSE_META_WORDS;
            let base_offset = load_i32(group_metadata, metadata);
            let set_count = load_i32(group_metadata, metadata + 1i32);
            let upper_offset = load_i32(group_metadata, metadata + 2i32);
            let transition_count = load_i32(group_metadata, metadata + 3i32);
            let transition_mask_offset = load_i32(group_metadata, metadata + 4i32);
            let gap_threshold_offset = load_i32(group_metadata, metadata + 5i32);
            let gap_search_steps = load_i32(group_metadata, metadata + 6i32);
            let key = seed ^ load_u64(group_keys, group);
            let mut state = mix_random(
                xori(shot_global, broadcast_scalar(key, const_shape![64])),
                rng_multiplier0,
                rng_multiplier1,
            );
            let mut position: Tile<i32, { [64] }> = zero_i32;
            let mut active: Tile<i32, { [64] }> = select(live, one_i32, zero_i32);
            loop {
                state = state + broadcast_scalar(rng_gamma, const_shape![64]);
                let bits = sparse_bits(state, rng_multiplier0, rng_multiplier1);
                let numerator = bits + bits + one_u64;
                let mut low = zero_i32;
                let mut high = broadcast_scalar(set_count, const_shape![64]);
                for _ in 0i32..gap_search_steps {
                    let searching = lt_tile(low, high);
                    let mid = shri(low + high, broadcast_scalar(1i32, const_shape![64]));
                    let safe_mid = select(searching, mid, zero_i32);
                    let threshold = load_u64_offsets(
                        gap_thresholds,
                        safe_mid + broadcast_scalar(gap_threshold_offset, const_shape![64]),
                        live,
                    );
                    let after = lt_tile(threshold, numerator);
                    high = select(searching, select(after, mid, high), high);
                    low = select(searching, select(after, low, mid + one_i32), low);
                }
                let gap = low;
                let candidate = position + gap;
                let set_count_i32: Tile<i32, { [64] }> =
                    broadcast_scalar(set_count, const_shape![64]);
                let within: Tile<i32, { [64] }> =
                    select(lt_tile(candidate, set_count_i32), one_i32, zero_i32);
                active = andi(active, within);
                let active_bool = ne_tile(active, zero_i32);
                let safe_set = select(active_bool, candidate, zero_i32);

                state = state + broadcast_scalar(rng_gamma, const_shape![64]);
                let outcome_uniform = sparse_uniform(state, rng_multiplier0, rng_multiplier1);
                let base_offsets = (safe_set + broadcast_scalar(base_offset, const_shape![64]))
                    * broadcast_scalar(mask_words, const_shape![64])
                    + broadcast_scalar(word, const_shape![64]);
                let mut event_mask = load_u64_offsets(base_masks, base_offsets, live);
                for transition in 0i32..transition_count {
                    let upper: Tile<f32, { [64] }> = broadcast_scalar(
                        load_f32(transition_upper, upper_offset + transition),
                        const_shape![64],
                    );
                    let hit = le_tile(outcome_uniform, upper);
                    let mask_offsets = (safe_set
                        * broadcast_scalar(transition_count, const_shape![64])
                        + broadcast_scalar(transition_mask_offset + transition, const_shape![64]))
                        * broadcast_scalar(mask_words, const_shape![64])
                        + broadcast_scalar(word, const_shape![64]);
                    let transition_mask = load_u64_offsets(transition_masks, mask_offsets, live);
                    event_mask = xori(event_mask, select(hit, transition_mask, zero_u64));
                }
                value = xori(value, select(active_bool, event_mask, zero_u64));
                position = select(active_bool, candidate + one_i32, set_count_i32);

                let active_count: Tile<i32, { [] }> = reduce_sum(active, 0i32);
                let active_count: i32 = tile_to_scalar(active_count);
                if active_count == 0i32 {
                    break;
                }
            }
        }
        store_u64_row(values, word, stride, shot_start, lanes, live, value);
    }

    #[cutile::entry()]
    unsafe fn wide_sample_init(
        state_re: *mut f32,
        state_im: *mut f32,
        branches: *mut u64,
        discarded: *mut u64,
        state_stride: i32,
        initial_dimension: i32,
        shots: i32,
    ) {
        let shot = get_tile_block_id().0;
        let lanes: Tile<i32, { [1024] }> = iota(const_shape![WIDE_TILE]);
        let zero: Tile<f32, { [1024] }> = constant(0.0f32, const_shape![WIDE_TILE]);
        let one: Tile<f32, { [1024] }> = constant(1.0f32, const_shape![WIDE_TILE]);
        let zero_i32: Tile<i32, { [1024] }> = constant(0i32, const_shape![WIDE_TILE]);
        let mut start = 0i32;
        while start < initial_dimension {
            let indices = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
            let live = lt_tile(
                indices,
                broadcast_scalar(initial_dimension, const_shape![WIDE_TILE]),
            );
            let offsets = indices + broadcast_scalar(shot * state_stride, const_shape![WIDE_TILE]);
            let basis_zero = eq_tile(indices, zero_i32);
            store_wide_f32(state_re, offsets, live, select(basis_zero, one, zero));
            store_wide_f32(state_im, offsets, live, zero);
            start = start + WIDE_TILE;
        }
        let zero_scalar: Tile<u64, { [] }> = scalar_to_tile(0u64);
        for word in 0i32..4i32 {
            store_u64(branches, word * shots + shot, zero_scalar);
        }
        store_u64(discarded, shot, zero_scalar);
    }

    #[cutile::entry()]
    #[allow(unused_assignments)]
    unsafe fn wide_sample_step(
        input_re: *mut f32,
        input_im: *mut f32,
        output_re: *mut f32,
        output_im: *mut f32,
        branches: *mut u64,
        discarded: *mut u64,
        metadata: *mut u64,
        controls: *mut i32,
        parameters: *mut f32,
        expression_values: *mut u64,
        randoms: *mut f32,
        instruction: i32,
        active_k: i32,
        input_stride: i32,
        output_stride: i32,
        shots: i32,
    ) {
        let shot = get_tile_block_id().0;
        let lanes1: Tile<i32, { [1] }> = iota(const_shape![1]);
        let live1: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let zero_u64_1: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let one_u64_1: Tile<u64, { [1] }> = constant(1u64, const_shape![1]);
        let half_f32_1: Tile<f32, { [1] }> = constant(0.5f32, const_shape![1]);
        let one_f32_1: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_floor_1: Tile<f32, { [1] }> = constant(1.0e-20f32, const_shape![1]);
        let branches0 = load_u64_row(branches, 0i32, shots, shot, lanes1, live1);
        let branches1 = load_u64_row(branches, 1i32, shots, shot, lanes1, live1);
        let branches2 = load_u64_row(branches, 2i32, shots, shot, lanes1, live1);
        let branches3 = load_u64_row(branches, 3i32, shots, shot, lanes1, live1);
        let rejected = load_u64(discarded, shot);

        if rejected == 0u64 {
            let meta = instruction * META_WORDS;
            let params = instruction * PARAM_WORDS;
            let opcode = load_u64(metadata, meta);
            let dimension = wide_dimension(active_k);
            let lanes: Tile<i32, { [1024] }> = iota(const_shape![WIDE_TILE]);
            let lanes_u64: Tile<u64, { [1024] }> = iota(const_shape![WIDE_TILE]);
            let zero_u64: Tile<u64, { [1024] }> = constant(0u64, const_shape![WIDE_TILE]);
            let zero_f32: Tile<f32, { [1024] }> = constant(0.0f32, const_shape![WIDE_TILE]);
            let one_f32: Tile<f32, { [1024] }> = constant(1.0f32, const_shape![WIDE_TILE]);
            let negative_one_f32: Tile<f32, { [1024] }> =
                constant(-1.0f32, const_shape![WIDE_TILE]);

            if opcode == 0u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    shots,
                    shot,
                    lanes1,
                    live1,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let sign = sign.broadcast(const_shape![WIDE_TILE]);
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![WIDE_TILE]);
                let coefficient_re =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![WIDE_TILE]);
                let coefficient_im =
                    broadcast_scalar(load_f32(parameters, params + 2i32), const_shape![WIDE_TILE]);
                let mut start = 0i32;
                let mut start_u64 = 0u64;
                if xmask == 0u64 {
                    while start < dimension {
                        let indices = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
                        let live = lt_tile(
                            indices,
                            broadcast_scalar(dimension, const_shape![WIDE_TILE]),
                        );
                        let basis =
                            lanes_u64 + broadcast_scalar(start_u64, const_shape![WIDE_TILE]);
                        let offsets = indices
                            + broadcast_scalar(shot * input_stride, const_shape![WIDE_TILE]);
                        let old_re = load_wide_f32(input_re, offsets, live);
                        let old_im = load_wide_f32(input_im, offsets, live);
                        let odd = ne_tile(
                            wide_parity(andi(
                                basis,
                                broadcast_scalar(zmask, const_shape![WIDE_TILE]),
                            )),
                            zero_u64,
                        );
                        let direction = select(ne_tile(sign, odd), negative_one_f32, one_f32);
                        let cr = direction * coefficient_re;
                        let ci = direction * coefficient_im;
                        store_wide_f32(
                            output_re,
                            offsets,
                            live,
                            c * old_re + cr * old_re - ci * old_im,
                        );
                        store_wide_f32(
                            output_im,
                            offsets,
                            live,
                            c * old_im + cr * old_im + ci * old_re,
                        );
                        start = start + WIDE_TILE;
                        start_u64 = start_u64 + 1024u64;
                    }
                } else {
                    let pair_count = dimension / 2i32;
                    let pair_bit = load_u64(metadata, meta + 7i32);
                    let uniform_q = select(sign, negative_one_f32 * coefficient_im, coefficient_im);
                    while start < pair_count {
                        let packed_i32 = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
                        let live = lt_tile(
                            packed_i32,
                            broadcast_scalar(pair_count, const_shape![WIDE_TILE]),
                        );
                        let packed =
                            lanes_u64 + broadcast_scalar(start_u64, const_shape![WIDE_TILE]);
                        let left = wide_insert_zero_bit(packed, pair_bit);
                        let right = xori(left, broadcast_scalar(xmask, const_shape![WIDE_TILE]));
                        let base = shot * input_stride;
                        let left_re = load_wide_f32_u64(input_re, base, left, live);
                        let left_im = load_wide_f32_u64(input_im, base, left, live);
                        let right_re = load_wide_f32_u64(input_re, base, right, live);
                        let right_im = load_wide_f32_u64(input_im, base, right, live);
                        let output_base = shot * output_stride;
                        if zmask == 0u64 {
                            store_wide_f32_u64(
                                output_re,
                                output_base,
                                left,
                                live,
                                c * left_re - uniform_q * right_im,
                            );
                            store_wide_f32_u64(
                                output_im,
                                output_base,
                                left,
                                live,
                                c * left_im + uniform_q * right_re,
                            );
                            store_wide_f32_u64(
                                output_re,
                                output_base,
                                right,
                                live,
                                c * right_re - uniform_q * left_im,
                            );
                            store_wide_f32_u64(
                                output_im,
                                output_base,
                                right,
                                live,
                                c * right_im + uniform_q * left_re,
                            );
                        } else {
                            let left_odd = ne_tile(
                                wide_parity(andi(
                                    left,
                                    broadcast_scalar(zmask, const_shape![WIDE_TILE]),
                                )),
                                zero_u64,
                            );
                            let right_odd = ne_tile(
                                wide_parity(andi(
                                    right,
                                    broadcast_scalar(zmask, const_shape![WIDE_TILE]),
                                )),
                                zero_u64,
                            );
                            let left_direction =
                                select(ne_tile(sign, left_odd), negative_one_f32, one_f32);
                            let right_direction =
                                select(ne_tile(sign, right_odd), negative_one_f32, one_f32);
                            let left_cr = left_direction * coefficient_re;
                            let left_ci = left_direction * coefficient_im;
                            let right_cr = right_direction * coefficient_re;
                            let right_ci = right_direction * coefficient_im;
                            store_wide_f32_u64(
                                output_re,
                                output_base,
                                left,
                                live,
                                c * left_re + right_cr * right_re - right_ci * right_im,
                            );
                            store_wide_f32_u64(
                                output_im,
                                output_base,
                                left,
                                live,
                                c * left_im + right_cr * right_im + right_ci * right_re,
                            );
                            store_wide_f32_u64(
                                output_re,
                                output_base,
                                right,
                                live,
                                c * right_re + left_cr * left_re - left_ci * left_im,
                            );
                            store_wide_f32_u64(
                                output_im,
                                output_base,
                                right,
                                live,
                                c * right_im + left_cr * left_im + left_ci * left_re,
                            );
                        }
                        start = start + WIDE_TILE;
                        start_u64 = start_u64 + 1024u64;
                    }
                }
            } else if opcode == 1u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    shots,
                    shot,
                    lanes1,
                    live1,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                )
                .broadcast(const_shape![WIDE_TILE]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![WIDE_TILE]);
                let sin =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![WIDE_TILE]);
                let q = select(sign, sin, negative_one_f32 * sin);
                let mut start = 0i32;
                while start < dimension {
                    let indices = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
                    let live = lt_tile(
                        indices,
                        broadcast_scalar(dimension, const_shape![WIDE_TILE]),
                    );
                    let input_offsets =
                        indices + broadcast_scalar(shot * input_stride, const_shape![WIDE_TILE]);
                    let low_offsets =
                        indices + broadcast_scalar(shot * output_stride, const_shape![WIDE_TILE]);
                    let high_offsets =
                        low_offsets + broadcast_scalar(dimension, const_shape![WIDE_TILE]);
                    let old_re = load_wide_f32(input_re, input_offsets, live);
                    let old_im = load_wide_f32(input_im, input_offsets, live);
                    store_wide_f32(output_re, low_offsets, live, c * old_re);
                    store_wide_f32(output_im, low_offsets, live, c * old_im);
                    store_wide_f32(output_re, high_offsets, live, negative_one_f32 * q * old_im);
                    store_wide_f32(output_im, high_offsets, live, q * old_re);
                    start = start + WIDE_TILE;
                }
            } else if opcode == 2u64 {
                let out_dimension = dimension / 2i32;
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let pivot = load_u64(metadata, meta + 7i32);
                let pivot_mask = wide_bit_mask(pivot);
                let diagonal_phase = load_u64(metadata, meta + 8i32) != 0u64;
                let z_without_pivot = load_u64(metadata, meta + 9i32);
                let coefficient_re =
                    broadcast_scalar(load_f32(parameters, params), const_shape![WIDE_TILE]);
                let coefficient_im =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![WIDE_TILE]);
                let inv_sqrt2 = broadcast_scalar(INV_SQRT2, const_shape![WIDE_TILE]);
                let mut probability_true: Tile<f32, { [] }> = scalar_to_tile(0.0f32);
                let mut start = 0i32;
                let mut start_u64 = 0u64;
                while start < out_dimension {
                    let packed_i32 = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
                    let live = lt_tile(
                        packed_i32,
                        broadcast_scalar(out_dimension, const_shape![WIDE_TILE]),
                    );
                    let packed = lanes_u64 + broadcast_scalar(start_u64, const_shape![WIDE_TILE]);
                    let source0 = wide_insert_zero_bit(packed, pivot);
                    let mut value_re = zero_f32;
                    let mut value_im = zero_f32;
                    if xmask == 0u64 {
                        let odd = ne_tile(
                            wide_parity(andi(
                                source0,
                                broadcast_scalar(z_without_pivot, const_shape![WIDE_TILE]),
                            )),
                            zero_u64,
                        );
                        let take_high = eq_tile(
                            odd,
                            broadcast_scalar(diagonal_phase, const_shape![WIDE_TILE]),
                        );
                        let source = ori(
                            source0,
                            select(
                                take_high,
                                broadcast_scalar(pivot_mask, const_shape![WIDE_TILE]),
                                zero_u64,
                            ),
                        );
                        let base = shot * input_stride;
                        value_re = load_wide_f32_u64(input_re, base, source, live);
                        value_im = load_wide_f32_u64(input_im, base, source, live);
                    } else {
                        let source1 =
                            xori(source0, broadcast_scalar(xmask, const_shape![WIDE_TILE]));
                        let base = shot * input_stride;
                        let re0 = load_wide_f32_u64(input_re, base, source0, live);
                        let im0 = load_wide_f32_u64(input_im, base, source0, live);
                        let re1 = load_wide_f32_u64(input_re, base, source1, live);
                        let im1 = load_wide_f32_u64(input_im, base, source1, live);
                        let odd = ne_tile(
                            wide_parity(andi(
                                source0,
                                broadcast_scalar(zmask, const_shape![WIDE_TILE]),
                            )),
                            zero_u64,
                        );
                        let direction = select(odd, one_f32, negative_one_f32);
                        let cr = direction * coefficient_re;
                        let ci = direction * coefficient_im;
                        value_re = inv_sqrt2 * re0 + cr * re1 - ci * im1;
                        value_im = inv_sqrt2 * im0 + cr * im1 + ci * re1;
                    }
                    let mass = select(live, value_re * value_re + value_im * value_im, zero_f32);
                    let partial: Tile<f32, { [] }> = reduce_sum(mass, 0i32);
                    probability_true = probability_true + partial;
                    start = start + WIDE_TILE;
                    start_u64 = start_u64 + 1024u64;
                }
                let zero_probability: Tile<f32, { [] }> = scalar_to_tile(0.0f32);
                let one_probability: Tile<f32, { [] }> = scalar_to_tile(1.0f32);
                let probability_true: Tile<f32, { [] }> = maxf(
                    probability_true,
                    zero_probability,
                    nan::Enabled,
                    ftz::Disabled,
                );
                let probability_true: Tile<f32, { [] }> = minf(
                    probability_true,
                    one_probability,
                    nan::Enabled,
                    ftz::Disabled,
                );
                let probability_true: Tile<f32, { [1] }> =
                    probability_true.reshape(const_shape![1]);
                let random_row = load_i32(controls, instruction);
                let uniform = load_randoms(randoms, random_row, shots, shot, lanes1, live1);
                let branch = lt_tile(uniform, probability_true);
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                store_wide_branch(
                    branches,
                    shot,
                    shots,
                    branch_slot,
                    branch_bit,
                    branch,
                    lanes1,
                    live1,
                );
                let probability: Tile<f32, { [1] }> =
                    select(branch, probability_true, one_f32_1 - probability_true);
                let probability: Tile<f32, { [1] }> = maxf(
                    probability,
                    probability_floor_1,
                    nan::Enabled,
                    ftz::Disabled,
                );
                let invnorm: Tile<f32, { [1] }> = rsqrt(probability, ftz::Disabled);
                let invnorm: Tile<f32, { [1024] }> = invnorm.broadcast(const_shape![WIDE_TILE]);
                let branch = branch.broadcast(const_shape![WIDE_TILE]);
                start = 0i32;
                start_u64 = 0u64;
                while start < out_dimension {
                    let packed_i32 = lanes + broadcast_scalar(start, const_shape![WIDE_TILE]);
                    let live = lt_tile(
                        packed_i32,
                        broadcast_scalar(out_dimension, const_shape![WIDE_TILE]),
                    );
                    let packed = lanes_u64 + broadcast_scalar(start_u64, const_shape![WIDE_TILE]);
                    let source0 = wide_insert_zero_bit(packed, pivot);
                    let mut value_re = zero_f32;
                    let mut value_im = zero_f32;
                    if xmask == 0u64 {
                        let odd = ne_tile(
                            wide_parity(andi(
                                source0,
                                broadcast_scalar(z_without_pivot, const_shape![WIDE_TILE]),
                            )),
                            zero_u64,
                        );
                        let false_high = ne_tile(
                            odd,
                            broadcast_scalar(diagonal_phase, const_shape![WIDE_TILE]),
                        );
                        let take_high = ne_tile(false_high, branch);
                        let source = ori(
                            source0,
                            select(
                                take_high,
                                broadcast_scalar(pivot_mask, const_shape![WIDE_TILE]),
                                zero_u64,
                            ),
                        );
                        let base = shot * input_stride;
                        value_re = load_wide_f32_u64(input_re, base, source, live);
                        value_im = load_wide_f32_u64(input_im, base, source, live);
                    } else {
                        let source1 =
                            xori(source0, broadcast_scalar(xmask, const_shape![WIDE_TILE]));
                        let base = shot * input_stride;
                        let re0 = load_wide_f32_u64(input_re, base, source0, live);
                        let im0 = load_wide_f32_u64(input_im, base, source0, live);
                        let re1 = load_wide_f32_u64(input_re, base, source1, live);
                        let im1 = load_wide_f32_u64(input_im, base, source1, live);
                        let odd = ne_tile(
                            wide_parity(andi(
                                source0,
                                broadcast_scalar(zmask, const_shape![WIDE_TILE]),
                            )),
                            zero_u64,
                        );
                        let direction = select(ne_tile(branch, odd), negative_one_f32, one_f32);
                        let cr = direction * coefficient_re;
                        let ci = direction * coefficient_im;
                        value_re = inv_sqrt2 * re0 + cr * re1 - ci * im1;
                        value_im = inv_sqrt2 * im0 + cr * im1 + ci * re1;
                    }
                    let output_offsets = packed_i32
                        + broadcast_scalar(shot * output_stride, const_shape![WIDE_TILE]);
                    store_wide_f32(output_re, output_offsets, live, value_re * invnorm);
                    store_wide_f32(output_im, output_offsets, live, value_im * invnorm);
                    start = start + WIDE_TILE;
                    start_u64 = start_u64 + 1024u64;
                }
            } else if opcode == 3u64 {
                let random_row = load_i32(controls, instruction);
                let uniform = load_randoms(randoms, random_row, shots, shot, lanes1, live1);
                let branch = lt_tile(uniform, half_f32_1);
                store_wide_branch(
                    branches,
                    shot,
                    shots,
                    load_u64(metadata, meta + 10i32),
                    load_u64(metadata, meta + 11i32),
                    branch,
                    lanes1,
                    live1,
                );
            } else {
                if load_u64(metadata, meta + 8i32) != 0u64 {
                    let outcome = instruction_expression_rows(
                        metadata,
                        controls,
                        expression_values,
                        shots,
                        shot,
                        lanes1,
                        live1,
                        instruction,
                        branches0,
                        branches1,
                        branches2,
                        branches3,
                    );
                    let value: Tile<u64, { [] }> =
                        reduce_sum(select(outcome, one_u64_1, zero_u64_1), 0i32);
                    store_u64(discarded, shot, value);
                }
            }
        }
    }

    #[cutile::entry()]
    unsafe fn wide_sample_finalize(
        block_counts: *mut u64,
        branches: *mut u64,
        discarded: *mut u64,
        expression_values: *mut u64,
        shots: i32,
        logical_word: i32,
        logical_block_mask: u64,
        logical_mask0: u64,
        logical_mask1: u64,
        logical_mask2: u64,
        logical_mask3: u64,
    ) {
        let shot = get_tile_block_id().0;
        let lanes: Tile<i32, { [1] }> = iota(const_shape![1]);
        let live: Tile<bool, { [1] }> = constant(true, const_shape![1]);
        let zero: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let one: Tile<u64, { [1] }> = constant(1u64, const_shape![1]);
        let branches0 = load_u64_row(branches, 0i32, shots, shot, lanes, live);
        let branches1 = load_u64_row(branches, 1i32, shots, shot, lanes, live);
        let branches2 = load_u64_row(branches, 2i32, shots, shot, lanes, live);
        let branches3 = load_u64_row(branches, 3i32, shots, shot, lanes, live);
        let logical_values =
            load_u64_row(expression_values, logical_word, shots, shot, lanes, live);
        let logical = expression(
            logical_values,
            logical_block_mask,
            logical_mask0,
            logical_mask1,
            logical_mask2,
            logical_mask3,
            branches0,
            branches1,
            branches2,
            branches3,
        );
        let discarded_value = load_u64(discarded, shot);
        let discarded_tile = broadcast_scalar(discarded_value, const_shape![1]);
        let accepted = eq_tile(discarded_tile, zero);
        let logical_value = select(accepted, select(logical, one, zero), zero);
        store_u64(block_counts, shot * 2i32, scalar_to_tile(discarded_value));
        let logical_value: Tile<u64, { [] }> = reduce_sum(logical_value, 0i32);
        store_u64(block_counts, shot * 2i32 + 1i32, logical_value);
    }

    #[cutile::entry()]
    unsafe fn sample16(
        block_counts: *mut u64,
        metadata: *mut u64,
        controls: *mut i32,
        parameters: *mut f32,
        expression_values: *mut u64,
        randoms: *mut f32,
        instruction_count: i32,
        random_stride: i32,
        shots: i32,
        logical_block_mask: u64,
        logical_mask0: u64,
        logical_mask1: u64,
        logical_mask2: u64,
        logical_mask3: u64,
    ) {
        let pid = get_tile_block_id().0;
        let shot_start = pid * 64i32;
        let lanes: Tile<i32, { [64] }> = iota(const_shape![64]);
        let shot_indices: Tile<i32, { [64] }> =
            lanes + broadcast_scalar(shot_start, const_shape![64]);
        let live = lt_tile(shot_indices, broadcast_scalar(shots, const_shape![64]));
        let exogenous_values: Tile<u64, { [64] }> = load_u64_row(
            expression_values,
            0i32,
            random_stride,
            shot_start,
            lanes,
            live,
        );

        let basis_1d: Tile<u64, { [16] }> = iota(const_shape![DIM]);
        let basis: Tile<u64, { [64, 16] }> = basis_1d
            .reshape(const_shape![1, DIM])
            .broadcast(const_shape![64, DIM]);
        let zero_state: Tile<f32, { [64, 16] }> = constant(0.0f32, const_shape![64, DIM]);
        let one_state: Tile<f32, { [64, 16] }> = constant(1.0f32, const_shape![64, DIM]);
        let negative_one_state: Tile<f32, { [64, 16] }> = constant(-1.0f32, const_shape![64, DIM]);
        let inv_sqrt2_state: Tile<f32, { [64, 16] }> = constant(INV_SQRT2, const_shape![64, DIM]);
        let zero_basis: Tile<u64, { [64, 16] }> = constant(0u64, const_shape![64, DIM]);
        let zero_shots: Tile<u64, { [64] }> = constant(0u64, const_shape![64]);
        let one_shots: Tile<u64, { [64] }> = constant(1u64, const_shape![64]);
        let zero_probability: Tile<f32, { [64] }> = constant(0.0f32, const_shape![64]);
        let half_probability: Tile<f32, { [64] }> = constant(0.5f32, const_shape![64]);
        let one_probability: Tile<f32, { [64] }> = constant(1.0f32, const_shape![64]);
        let probability_floor: Tile<f32, { [64] }> = constant(1.0e-20f32, const_shape![64]);
        let basis_zero = eq_tile(basis, zero_basis);
        let mut re = select(basis_zero, one_state, zero_state);
        let mut im = zero_state;
        let mut branches0 = zero_shots;
        let mut branches1 = zero_shots;
        let mut branches2 = zero_shots;
        let mut branches3 = zero_shots;
        let mut discarded = zero_shots;

        for instruction in 0i32..instruction_count {
            let meta = instruction * META_WORDS;
            let params = instruction * PARAM_WORDS;
            let control = instruction * CONTROL_WORDS;
            let opcode = load_u64(metadata, meta);

            if opcode == 0u64 {
                let sign = instruction_expression(
                    metadata,
                    exogenous_values,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let partner_re = flip_mask(re, xmask);
                let partner_im = flip_mask(im, xmask);
                let sign_2d = sign
                    .reshape(const_shape![64, 1])
                    .broadcast(const_shape![64, DIM]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![64, DIM]);
                let old_re = re;
                let old_im = im;
                if zmask == 0u64 {
                    let direction = select(sign_2d, negative_one_state, one_state);
                    let q = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 2i32),
                            const_shape![64, DIM],
                        );
                    re = c * old_re - q * partner_im;
                    im = c * old_im + q * partner_re;
                } else {
                    let partner_basis = xori(basis, broadcast_scalar(xmask, const_shape![64, DIM]));
                    let odd = ne_tile(
                        parity(andi(
                            partner_basis,
                            broadcast_scalar(zmask, const_shape![64, DIM]),
                        )),
                        zero_basis,
                    );
                    let direction = select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                    let cr = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![64, DIM],
                        );
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 2i32),
                            const_shape![64, DIM],
                        );
                    re = c * old_re + cr * partner_re - ci * partner_im;
                    im = c * old_im + cr * partner_im + ci * partner_re;
                }
            } else if opcode == 1u64 {
                let sign = instruction_expression(
                    metadata,
                    exogenous_values,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let sign_2d = sign
                    .reshape(const_shape![64, 1])
                    .broadcast(const_shape![64, DIM]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![64, DIM]);
                let sin =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![64, DIM]);
                // CPU promotion uses q=-sin for false and q=+sin for true.
                let q = select(sign_2d, sin, negative_one_state * sin);
                let physical_bit = load_u64(metadata, meta + 7i32);
                let copy_re = flip_dynamic(re, physical_bit);
                let copy_im = flip_dynamic(im, physical_bit);
                let physical_clear = eq_tile(
                    andi(
                        basis,
                        broadcast_scalar(bit_mask(physical_bit), const_shape![64, DIM]),
                    ),
                    zero_basis,
                );
                re = select(physical_clear, c * re, negative_one_state * q * copy_im);
                im = select(physical_clear, c * im, q * copy_re);
            } else if opcode == 2u64 {
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let pivot = load_u64(metadata, meta + 7i32);
                let diagonal_phase_word = load_u64(metadata, meta + 8i32);
                let diagonal_phase = diagonal_phase_word != 0u64;
                let z_without = load_u64(metadata, meta + 9i32);
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let diagonal = xmask == 0u64;
                let pivot_bit = bit_mask(pivot);
                let mut probability_true = zero_probability;
                if diagonal {
                    let odd = ne_tile(
                        parity(andi(basis, broadcast_scalar(zmask, const_shape![64, DIM]))),
                        zero_basis,
                    );
                    let target = diagonal_phase_word == 0u64;
                    let selected = eq_tile(odd, broadcast_scalar(target, const_shape![64, DIM]));
                    let mass = select(selected, re * re + im * im, zero_state);
                    probability_true = reduce_sum(mass, 1i32);
                }
                if xmask != 0u64 {
                    let partner_re = flip_mask(re, xmask);
                    let partner_im = flip_mask(im, xmask);
                    let odd = ne_tile(
                        parity(andi(basis, broadcast_scalar(zmask, const_shape![64, DIM]))),
                        zero_basis,
                    );
                    let direction = select(odd, one_state, negative_one_state);
                    let cr = direction
                        * broadcast_scalar(load_f32(parameters, params), const_shape![64, DIM]);
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![64, DIM],
                        );
                    let ar = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
                    let ai = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
                    let pivot_clear = eq_tile(
                        andi(basis, broadcast_scalar(pivot_bit, const_shape![64, DIM])),
                        zero_basis,
                    );
                    let mass = select(pivot_clear, ar * ar + ai * ai, zero_state);
                    probability_true = reduce_sum(mass, 1i32);
                }
                probability_true = minf(
                    maxf(
                        probability_true,
                        zero_probability,
                        nan::Enabled,
                        ftz::Disabled,
                    ),
                    one_probability,
                    nan::Enabled,
                    ftz::Disabled,
                );
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, probability_true);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![64]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }

                let branch_2d = branch
                    .reshape(const_shape![64, 1])
                    .broadcast(const_shape![64, DIM]);
                let probability =
                    select(branch, probability_true, one_probability - probability_true);
                let invnorm = rsqrt(
                    maxf(probability, probability_floor, nan::Enabled, ftz::Disabled),
                    ftz::Disabled,
                )
                .reshape(const_shape![64, 1])
                .broadcast(const_shape![64, DIM]);
                let pivot_clear = eq_tile(
                    andi(basis, broadcast_scalar(pivot_bit, const_shape![64, DIM])),
                    zero_basis,
                );
                let mut out_re = zero_state;
                let mut out_im = zero_state;
                if diagonal {
                    let partner_re = flip_dynamic(re, pivot);
                    let partner_im = flip_dynamic(im, pivot);
                    let odd_bits = parity(andi(
                        basis,
                        broadcast_scalar(z_without, const_shape![64, DIM]),
                    ));
                    let odd = ne_tile(odd_bits, zero_basis);
                    let phase: Tile<bool, { [64, 16] }> =
                        broadcast_scalar(diagonal_phase, const_shape![64, DIM]);
                    let odd_branch: Tile<bool, { [64, 16] }> = ne_tile(odd, branch_2d);
                    let take_high = ne_tile(odd_branch, phase);
                    out_re = select(take_high, partner_re, re);
                    out_im = select(take_high, partner_im, im);
                }
                if xmask != 0u64 {
                    let partner_re = flip_mask(re, xmask);
                    let partner_im = flip_mask(im, xmask);
                    let odd = ne_tile(
                        parity(andi(basis, broadcast_scalar(zmask, const_shape![64, DIM]))),
                        zero_basis,
                    );
                    let negative = ne_tile(branch_2d, odd);
                    let direction = select(negative, negative_one_state, one_state);
                    let cr = direction
                        * broadcast_scalar(load_f32(parameters, params), const_shape![64, DIM]);
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![64, DIM],
                        );
                    out_re = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
                    out_im = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
                }
                out_re = select(pivot_clear, out_re, zero_state);
                out_im = select(pivot_clear, out_im, zero_state);
                re = out_re * invnorm;
                im = out_im * invnorm;
            } else if opcode == 3u64 {
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, half_probability);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![64]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }
            } else {
                let outcome = instruction_expression(
                    metadata,
                    exogenous_values,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                if load_u64(metadata, meta + 8i32) != 0u64 {
                    discarded = ori(discarded, select(outcome, one_shots, zero_shots));
                }
            }
        }

        let logical = expression(
            exogenous_values,
            logical_block_mask,
            logical_mask0,
            logical_mask1,
            logical_mask2,
            logical_mask3,
            branches0,
            branches1,
            branches2,
            branches3,
        );
        let live_bits = select(live, one_shots, zero_shots);
        let discarded_bits = andi(discarded, live_bits);
        let logical_bits = select(logical, one_shots, zero_shots);
        let accepted = eq_tile(discarded_bits, zero_shots);
        let logical_error_bits = andi(select(accepted, logical_bits, zero_shots), live_bits);
        let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded_bits, 0i32);
        let logical_error_count: Tile<u64, { [] }> = reduce_sum(logical_error_bits, 0i32);
        store_u64(block_counts, pid * 2i32, discarded_count);
        store_u64(block_counts, pid * 2i32 + 1i32, logical_error_count);
    }

    #[cutile::entry()]
    unsafe fn sample128(
        block_counts: *mut u64,
        metadata: *mut u64,
        controls: *mut i32,
        expectation_indices: *mut i32,
        parameters: *mut f32,
        expression_values: *mut u64,
        expectations: *mut f32,
        detectors: *mut u64,
        randoms: *mut f32,
        instruction_count: i32,
        random_stride: i32,
        shots: i32,
        keep_records: i32,
        logical_word: i32,
        logical_block_mask: u64,
        logical_mask0: u64,
        logical_mask1: u64,
        logical_mask2: u64,
        logical_mask3: u64,
    ) {
        let pid = get_tile_block_id().0;
        let shot_start = pid;
        let lanes: Tile<i32, { [1] }> = iota(const_shape![1]);
        let shot_indices: Tile<i32, { [1] }> =
            lanes + broadcast_scalar(shot_start, const_shape![1]);
        let live = lt_tile(shot_indices, broadcast_scalar(shots, const_shape![1]));
        let basis_1d: Tile<u64, { [128] }> = iota(const_shape![128]);
        let basis: Tile<u64, { [1, 128] }> = basis_1d
            .reshape(const_shape![1, 128])
            .broadcast(const_shape![1, 128]);
        let zero_state: Tile<f32, { [1, 128] }> = constant(0.0f32, const_shape![1, 128]);
        let one_state: Tile<f32, { [1, 128] }> = constant(1.0f32, const_shape![1, 128]);
        let negative_one_state: Tile<f32, { [1, 128] }> = constant(-1.0f32, const_shape![1, 128]);
        let inv_sqrt2_state: Tile<f32, { [1, 128] }> = constant(INV_SQRT2, const_shape![1, 128]);
        let zero_basis: Tile<u64, { [1, 128] }> = constant(0u64, const_shape![1, 128]);
        let zero_shots: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let one_shots: Tile<u64, { [1] }> = constant(1u64, const_shape![1]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let half_probability: Tile<f32, { [1] }> = constant(0.5f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_floor: Tile<f32, { [1] }> = constant(1.0e-20f32, const_shape![1]);
        let basis_zero = eq_tile(basis, zero_basis);
        let mut re = select(basis_zero, one_state, zero_state);
        let mut im = zero_state;
        let mut branches0 = zero_shots;
        let mut branches1 = zero_shots;
        let mut branches2 = zero_shots;
        let mut branches3 = zero_shots;
        let mut discarded = zero_shots;

        let mut instruction = 0i32;
        while instruction < instruction_count {
            let meta = instruction * META_WORDS;
            let params = instruction * PARAM_WORDS;
            let control = instruction * CONTROL_WORDS;
            let opcode = load_u64(metadata, meta);

            if opcode == 0u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let x_basis_mask = load_u64(metadata, meta + 7i32);
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 128]);
                if x_basis_mask != 0u64 && load_u64(metadata, meta + 8i32) != 0u64 {
                    let scale =
                        broadcast_scalar(load_f32(parameters, params + 3i32), const_shape![1, 128]);
                    re = hadamard_mask_compact(re, x_basis_mask) * scale;
                    im = hadamard_mask_compact(im, x_basis_mask) * scale;
                }
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 128]);
                let old_re = re;
                let old_im = im;
                if x_basis_mask != 0u64 {
                    // X-only Paulis are diagonal between the two run-boundary
                    // Hadamard transforms, so no state permutation is needed.
                    let odd = ne_tile(
                        parity_compact(andi(basis, broadcast_scalar(xmask, const_shape![1, 128]))),
                        zero_basis,
                    );
                    let direction = select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                    let q = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 2i32),
                            const_shape![1, 128],
                        );
                    re = c * old_re - q * old_im;
                    im = c * old_im + q * old_re;
                    if load_u64(metadata, meta + 9i32) != 0u64 {
                        let scale = broadcast_scalar(
                            load_f32(parameters, params + 3i32),
                            const_shape![1, 128],
                        );
                        re = hadamard_mask_compact(re, x_basis_mask) * scale;
                        im = hadamard_mask_compact(im, x_basis_mask) * scale;
                    }
                } else {
                    let partner_re = flip_mask_compact(re, xmask);
                    let partner_im = flip_mask_compact(im, xmask);
                    if zmask == 0u64 {
                        let direction = select(sign_2d, negative_one_state, one_state);
                        let q = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 128],
                            );
                        re = c * old_re - q * partner_im;
                        im = c * old_im + q * partner_re;
                    } else {
                        let partner_basis =
                            xori(basis, broadcast_scalar(xmask, const_shape![1, 128]));
                        let odd = ne_tile(
                            parity_compact(andi(
                                partner_basis,
                                broadcast_scalar(zmask, const_shape![1, 128]),
                            )),
                            zero_basis,
                        );
                        let direction =
                            select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                        let cr = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 1i32),
                                const_shape![1, 128],
                            );
                        let ci = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 128],
                            );
                        re = c * old_re + cr * partner_re - ci * partner_im;
                        im = c * old_im + cr * partner_im + ci * partner_re;
                    }
                }
            } else if opcode == 1u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 128]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 128]);
                let sin =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 128]);
                // CPU promotion uses q=-sin for false and q=+sin for true.
                let q = select(sign_2d, sin, negative_one_state * sin);
                let physical_bit = load_u64(metadata, meta + 7i32);
                let copy_re = flip_dynamic_compact(re, physical_bit);
                let copy_im = flip_dynamic_compact(im, physical_bit);
                let physical_clear = eq_tile(
                    andi(
                        basis,
                        broadcast_scalar(bit_mask(physical_bit), const_shape![1, 128]),
                    ),
                    zero_basis,
                );
                re = select(physical_clear, c * re, negative_one_state * q * copy_im);
                im = select(physical_clear, c * im, q * copy_re);
            } else if opcode == 2u64 {
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let pivot = load_u64(metadata, meta + 7i32);
                let diagonal_phase_word = load_u64(metadata, meta + 8i32);
                let diagonal_phase = diagonal_phase_word != 0u64;
                let z_without = load_u64(metadata, meta + 9i32);
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let diagonal = xmask == 0u64;
                let pivot_bit = bit_mask(pivot);
                let probability_true = measurement_probability_compact(
                    re,
                    im,
                    basis,
                    parameters,
                    params,
                    xmask,
                    zmask,
                    pivot,
                    diagonal_phase_word,
                );
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, probability_true);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }

                let branch_2d = branch
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 128]);
                let probability =
                    select(branch, probability_true, one_probability - probability_true);
                let invnorm = rsqrt(
                    maxf(probability, probability_floor, nan::Enabled, ftz::Disabled),
                    ftz::Disabled,
                )
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 128]);
                let pivot_clear = eq_tile(
                    andi(basis, broadcast_scalar(pivot_bit, const_shape![1, 128])),
                    zero_basis,
                );
                let mut out_re = zero_state;
                let mut out_im = zero_state;
                if diagonal {
                    let partner_re = flip_dynamic_compact(re, pivot);
                    let partner_im = flip_dynamic_compact(im, pivot);
                    let odd_bits = parity_compact(andi(
                        basis,
                        broadcast_scalar(z_without, const_shape![1, 128]),
                    ));
                    let odd = ne_tile(odd_bits, zero_basis);
                    let phase: Tile<bool, { [1, 128] }> =
                        broadcast_scalar(diagonal_phase, const_shape![1, 128]);
                    let odd_branch: Tile<bool, { [1, 128] }> = ne_tile(odd, branch_2d);
                    let take_high = ne_tile(odd_branch, phase);
                    out_re = select(take_high, partner_re, re);
                    out_im = select(take_high, partner_im, im);
                }
                if xmask != 0u64 {
                    let partner_re = flip_mask_compact(re, xmask);
                    let partner_im = flip_mask_compact(im, xmask);
                    let odd = ne_tile(
                        parity_compact(andi(basis, broadcast_scalar(zmask, const_shape![1, 128]))),
                        zero_basis,
                    );
                    let negative = ne_tile(branch_2d, odd);
                    let direction = select(negative, negative_one_state, one_state);
                    let cr = direction
                        * broadcast_scalar(load_f32(parameters, params), const_shape![1, 128]);
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![1, 128],
                        );
                    out_re = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
                    out_im = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
                }
                out_re = select(pivot_clear, out_re, zero_state);
                out_im = select(pivot_clear, out_im, zero_state);
                re = out_re * invnorm;
                im = out_im * invnorm;
            } else if opcode == 3u64 {
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, half_probability);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }
            } else if opcode == 5u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    select(sign, zero_probability - one_probability, one_probability),
                );
            } else if opcode == 6u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let probability_true = measurement_probability_compact(
                    re,
                    im,
                    basis,
                    parameters,
                    params,
                    load_u64(metadata, meta + 5i32),
                    load_u64(metadata, meta + 6i32),
                    load_u64(metadata, meta + 7i32),
                    load_u64(metadata, meta + 8i32),
                );
                let expectation = select(sign, zero_probability - one_probability, one_probability)
                    * (one_probability
                        - broadcast_scalar(2.0f32, const_shape![1]) * probability_true);
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    expectation,
                );
            } else {
                let outcome = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                if keep_records != 0i32 {
                    store_u64_row(
                        detectors,
                        load_i32(expectation_indices, instruction),
                        shots,
                        shot_start,
                        lanes,
                        live,
                        select(outcome, one_shots, zero_shots),
                    );
                }
                if load_u64(metadata, meta + 8i32) != 0u64 {
                    discarded = ori(discarded, select(outcome, one_shots, zero_shots));
                    // A rejected one-shot tile cannot contribute a logical
                    // error, so its later quantum state is unobservable.
                    let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded, 0i32);
                    let discarded_count: u64 = tile_to_scalar(discarded_count);
                    if discarded_count != 0u64 {
                        break;
                    }
                }
            }
            instruction = instruction + 1i32;
        }

        let logical_values = load_u64_row(
            expression_values,
            logical_word,
            random_stride,
            shot_start,
            lanes,
            live,
        );
        let logical = expression(
            logical_values,
            logical_block_mask,
            logical_mask0,
            logical_mask1,
            logical_mask2,
            logical_mask3,
            branches0,
            branches1,
            branches2,
            branches3,
        );
        let live_bits = select(live, one_shots, zero_shots);
        let discarded_bits = andi(discarded, live_bits);
        let logical_bits = select(logical, one_shots, zero_shots);
        let accepted = eq_tile(discarded_bits, zero_shots);
        let logical_error_bits = andi(select(accepted, logical_bits, zero_shots), live_bits);
        let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded_bits, 0i32);
        let logical_error_count: Tile<u64, { [] }> = reduce_sum(logical_error_bits, 0i32);
        store_u64(block_counts, pid * 2i32, discarded_count);
        store_u64(block_counts, pid * 2i32 + 1i32, logical_error_count);
    }

    #[cutile::entry()]
    unsafe fn sample1024(
        block_counts: *mut u64,
        metadata: *mut u64,
        controls: *mut i32,
        expectation_indices: *mut i32,
        parameters: *mut f32,
        expression_values: *mut u64,
        expectations: *mut f32,
        detectors: *mut u64,
        randoms: *mut f32,
        instruction_count: i32,
        random_stride: i32,
        shots: i32,
        keep_records: i32,
        logical_word: i32,
        logical_block_mask: u64,
        logical_mask0: u64,
        logical_mask1: u64,
        logical_mask2: u64,
        logical_mask3: u64,
    ) {
        let pid = get_tile_block_id().0;
        let shot_start = pid;
        let lanes: Tile<i32, { [1] }> = iota(const_shape![1]);
        let shot_indices: Tile<i32, { [1] }> =
            lanes + broadcast_scalar(shot_start, const_shape![1]);
        let live = lt_tile(shot_indices, broadcast_scalar(shots, const_shape![1]));
        let basis_1d: Tile<u64, { [1024] }> = iota(const_shape![1024]);
        let basis: Tile<u64, { [1, 1024] }> = basis_1d
            .reshape(const_shape![1, 1024])
            .broadcast(const_shape![1, 1024]);
        let zero_state: Tile<f32, { [1, 1024] }> = constant(0.0f32, const_shape![1, 1024]);
        let one_state: Tile<f32, { [1, 1024] }> = constant(1.0f32, const_shape![1, 1024]);
        let negative_one_state: Tile<f32, { [1, 1024] }> = constant(-1.0f32, const_shape![1, 1024]);
        let inv_sqrt2_state: Tile<f32, { [1, 1024] }> = constant(INV_SQRT2, const_shape![1, 1024]);
        let zero_basis: Tile<u64, { [1, 1024] }> = constant(0u64, const_shape![1, 1024]);
        let zero_shots: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let one_shots: Tile<u64, { [1] }> = constant(1u64, const_shape![1]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let half_probability: Tile<f32, { [1] }> = constant(0.5f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_floor: Tile<f32, { [1] }> = constant(1.0e-20f32, const_shape![1]);
        let basis_zero = eq_tile(basis, zero_basis);
        let mut re = select(basis_zero, one_state, zero_state);
        let mut im = zero_state;
        let mut branches0 = zero_shots;
        let mut branches1 = zero_shots;
        let mut branches2 = zero_shots;
        let mut branches3 = zero_shots;
        let mut discarded = zero_shots;

        let mut instruction = 0i32;
        while instruction < instruction_count {
            let meta = instruction * META_WORDS;
            let params = instruction * PARAM_WORDS;
            let control = instruction * CONTROL_WORDS;
            let opcode = load_u64(metadata, meta);

            if opcode == 0u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let x_basis_mask = load_u64(metadata, meta + 7i32);
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 1024]);
                if x_basis_mask != 0u64 && load_u64(metadata, meta + 8i32) != 0u64 {
                    let scale = broadcast_scalar(
                        load_f32(parameters, params + 3i32),
                        const_shape![1, 1024],
                    );
                    re = hadamard_mask_medium(re, x_basis_mask) * scale;
                    im = hadamard_mask_medium(im, x_basis_mask) * scale;
                }
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 1024]);
                let old_re = re;
                let old_im = im;
                if x_basis_mask != 0u64 {
                    // X-only Paulis are diagonal between the two run-boundary
                    // Hadamard transforms, so no state permutation is needed.
                    let odd = ne_tile(
                        parity_medium(andi(basis, broadcast_scalar(xmask, const_shape![1, 1024]))),
                        zero_basis,
                    );
                    let direction = select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                    let q = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 2i32),
                            const_shape![1, 1024],
                        );
                    re = c * old_re - q * old_im;
                    im = c * old_im + q * old_re;
                    if load_u64(metadata, meta + 9i32) != 0u64 {
                        let scale = broadcast_scalar(
                            load_f32(parameters, params + 3i32),
                            const_shape![1, 1024],
                        );
                        re = hadamard_mask_medium(re, x_basis_mask) * scale;
                        im = hadamard_mask_medium(im, x_basis_mask) * scale;
                    }
                } else {
                    let partner_re = flip_mask_medium(re, xmask);
                    let partner_im = flip_mask_medium(im, xmask);
                    if zmask == 0u64 {
                        let direction = select(sign_2d, negative_one_state, one_state);
                        let q = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 1024],
                            );
                        re = c * old_re - q * partner_im;
                        im = c * old_im + q * partner_re;
                    } else {
                        let partner_basis =
                            xori(basis, broadcast_scalar(xmask, const_shape![1, 1024]));
                        let odd = ne_tile(
                            parity_medium(andi(
                                partner_basis,
                                broadcast_scalar(zmask, const_shape![1, 1024]),
                            )),
                            zero_basis,
                        );
                        let direction =
                            select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                        let cr = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 1i32),
                                const_shape![1, 1024],
                            );
                        let ci = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 1024],
                            );
                        re = c * old_re + cr * partner_re - ci * partner_im;
                        im = c * old_im + cr * partner_im + ci * partner_re;
                    }
                }
            } else if opcode == 1u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 1024]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 1024]);
                let sin =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 1024]);
                // CPU promotion uses q=-sin for false and q=+sin for true.
                let q = select(sign_2d, sin, negative_one_state * sin);
                let physical_bit = load_u64(metadata, meta + 7i32);
                let copy_re = flip_dynamic_medium(re, physical_bit);
                let copy_im = flip_dynamic_medium(im, physical_bit);
                let physical_clear = eq_tile(
                    andi(
                        basis,
                        broadcast_scalar(bit_mask(physical_bit), const_shape![1, 1024]),
                    ),
                    zero_basis,
                );
                re = select(physical_clear, c * re, negative_one_state * q * copy_im);
                im = select(physical_clear, c * im, q * copy_re);
            } else if opcode == 2u64 {
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let pivot = load_u64(metadata, meta + 7i32);
                let diagonal_phase_word = load_u64(metadata, meta + 8i32);
                let diagonal_phase = diagonal_phase_word != 0u64;
                let z_without = load_u64(metadata, meta + 9i32);
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let diagonal = xmask == 0u64;
                let pivot_bit = bit_mask(pivot);
                let probability_true = measurement_probability_medium(
                    re,
                    im,
                    basis,
                    parameters,
                    params,
                    xmask,
                    zmask,
                    pivot,
                    diagonal_phase_word,
                );
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, probability_true);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }

                let branch_2d = branch
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 1024]);
                let probability =
                    select(branch, probability_true, one_probability - probability_true);
                let invnorm = rsqrt(
                    maxf(probability, probability_floor, nan::Enabled, ftz::Disabled),
                    ftz::Disabled,
                )
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 1024]);
                let pivot_clear = eq_tile(
                    andi(basis, broadcast_scalar(pivot_bit, const_shape![1, 1024])),
                    zero_basis,
                );
                let mut out_re = zero_state;
                let mut out_im = zero_state;
                if diagonal {
                    let partner_re = flip_dynamic_medium(re, pivot);
                    let partner_im = flip_dynamic_medium(im, pivot);
                    let odd_bits = parity_medium(andi(
                        basis,
                        broadcast_scalar(z_without, const_shape![1, 1024]),
                    ));
                    let odd = ne_tile(odd_bits, zero_basis);
                    let phase: Tile<bool, { [1, 1024] }> =
                        broadcast_scalar(diagonal_phase, const_shape![1, 1024]);
                    let odd_branch: Tile<bool, { [1, 1024] }> = ne_tile(odd, branch_2d);
                    let take_high = ne_tile(odd_branch, phase);
                    out_re = select(take_high, partner_re, re);
                    out_im = select(take_high, partner_im, im);
                }
                if xmask != 0u64 {
                    let partner_re = flip_mask_medium(re, xmask);
                    let partner_im = flip_mask_medium(im, xmask);
                    let odd = ne_tile(
                        parity_medium(andi(basis, broadcast_scalar(zmask, const_shape![1, 1024]))),
                        zero_basis,
                    );
                    let negative = ne_tile(branch_2d, odd);
                    let direction = select(negative, negative_one_state, one_state);
                    let cr = direction
                        * broadcast_scalar(load_f32(parameters, params), const_shape![1, 1024]);
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![1, 1024],
                        );
                    out_re = inv_sqrt2_state * re + cr * partner_re - ci * partner_im;
                    out_im = inv_sqrt2_state * im + cr * partner_im + ci * partner_re;
                }
                out_re = select(pivot_clear, out_re, zero_state);
                out_im = select(pivot_clear, out_im, zero_state);
                re = out_re * invnorm;
                im = out_im * invnorm;
            } else if opcode == 3u64 {
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, half_probability);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }
            } else if opcode == 5u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    select(sign, zero_probability - one_probability, one_probability),
                );
            } else if opcode == 6u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let probability_true = measurement_probability_medium(
                    re,
                    im,
                    basis,
                    parameters,
                    params,
                    load_u64(metadata, meta + 5i32),
                    load_u64(metadata, meta + 6i32),
                    load_u64(metadata, meta + 7i32),
                    load_u64(metadata, meta + 8i32),
                );
                let expectation = select(sign, zero_probability - one_probability, one_probability)
                    * (one_probability
                        - broadcast_scalar(2.0f32, const_shape![1]) * probability_true);
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    expectation,
                );
            } else {
                let outcome = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                if keep_records != 0i32 {
                    store_u64_row(
                        detectors,
                        load_i32(expectation_indices, instruction),
                        shots,
                        shot_start,
                        lanes,
                        live,
                        select(outcome, one_shots, zero_shots),
                    );
                }
                if load_u64(metadata, meta + 8i32) != 0u64 {
                    discarded = ori(discarded, select(outcome, one_shots, zero_shots));
                    // A rejected one-shot tile cannot contribute a logical
                    // error, so its later quantum state is unobservable.
                    let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded, 0i32);
                    let discarded_count: u64 = tile_to_scalar(discarded_count);
                    if discarded_count != 0u64 {
                        break;
                    }
                }
            }
            instruction = instruction + 1i32;
        }

        let logical_values = load_u64_row(
            expression_values,
            logical_word,
            random_stride,
            shot_start,
            lanes,
            live,
        );
        let logical = expression(
            logical_values,
            logical_block_mask,
            logical_mask0,
            logical_mask1,
            logical_mask2,
            logical_mask3,
            branches0,
            branches1,
            branches2,
            branches3,
        );
        let live_bits = select(live, one_shots, zero_shots);
        let discarded_bits = andi(discarded, live_bits);
        let logical_bits = select(logical, one_shots, zero_shots);
        let accepted = eq_tile(discarded_bits, zero_shots);
        let logical_error_bits = andi(select(accepted, logical_bits, zero_shots), live_bits);
        let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded_bits, 0i32);
        let logical_error_count: Tile<u64, { [] }> = reduce_sum(logical_error_bits, 0i32);
        store_u64(block_counts, pid * 2i32, discarded_count);
        store_u64(block_counts, pid * 2i32 + 1i32, logical_error_count);
    }

    #[cutile::entry()]
    unsafe fn sample4096(
        block_counts: *mut u64,
        metadata: *mut u64,
        controls: *mut i32,
        expectation_indices: *mut i32,
        parameters: *mut f32,
        expression_values: *mut u64,
        expectations: *mut f32,
        detectors: *mut u64,
        randoms: *mut f32,
        instruction_count: i32,
        random_stride: i32,
        shots: i32,
        keep_records: i32,
        logical_word: i32,
        logical_block_mask: u64,
        logical_mask0: u64,
        logical_mask1: u64,
        logical_mask2: u64,
        logical_mask3: u64,
    ) {
        let pid = get_tile_block_id().0;
        let shot_start = pid;
        let lanes: Tile<i32, { [1] }> = iota(const_shape![1]);
        let shot_indices: Tile<i32, { [1] }> =
            lanes + broadcast_scalar(shot_start, const_shape![1]);
        let live = lt_tile(shot_indices, broadcast_scalar(shots, const_shape![1]));
        let basis_1d: Tile<u64, { [4096] }> = iota(const_shape![4096]);
        let basis: Tile<u64, { [1, 4096] }> = basis_1d
            .reshape(const_shape![1, 4096])
            .broadcast(const_shape![1, 4096]);
        let zero_state: Tile<f32, { [1, 4096] }> = constant(0.0f32, const_shape![1, 4096]);
        let one_state: Tile<f32, { [1, 4096] }> = constant(1.0f32, const_shape![1, 4096]);
        let negative_one_state: Tile<f32, { [1, 4096] }> = constant(-1.0f32, const_shape![1, 4096]);
        let inv_sqrt2_state: Tile<f32, { [1, 4096] }> = constant(INV_SQRT2, const_shape![1, 4096]);
        let zero_basis: Tile<u64, { [1, 4096] }> = constant(0u64, const_shape![1, 4096]);
        let zero_shots: Tile<u64, { [1] }> = constant(0u64, const_shape![1]);
        let one_shots: Tile<u64, { [1] }> = constant(1u64, const_shape![1]);
        let zero_probability: Tile<f32, { [1] }> = constant(0.0f32, const_shape![1]);
        let half_probability: Tile<f32, { [1] }> = constant(0.5f32, const_shape![1]);
        let one_probability: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
        let probability_floor: Tile<f32, { [1] }> = constant(1.0e-20f32, const_shape![1]);
        let basis_zero = eq_tile(basis, zero_basis);
        let mut re = select(basis_zero, one_state, zero_state);
        let mut im = zero_state;
        let mut branches0 = zero_shots;
        let mut branches1 = zero_shots;
        let mut branches2 = zero_shots;
        let mut branches3 = zero_shots;
        let mut discarded = zero_shots;

        let mut instruction = 0i32;
        while instruction < instruction_count {
            let meta = instruction * META_WORDS;
            let params = instruction * PARAM_WORDS;
            let control = instruction * CONTROL_WORDS;
            let opcode = load_u64(metadata, meta);

            if opcode == 0u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let x_basis_mask = load_u64(metadata, meta + 7i32);
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 4096]);
                if x_basis_mask != 0u64 && load_u64(metadata, meta + 8i32) != 0u64 {
                    let scale = broadcast_scalar(
                        load_f32(parameters, params + 3i32),
                        const_shape![1, 4096],
                    );
                    re = hadamard_mask_large(re, x_basis_mask) * scale;
                    im = hadamard_mask_large(im, x_basis_mask) * scale;
                }
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 4096]);
                let old_re = re;
                let old_im = im;
                if x_basis_mask != 0u64 {
                    // X-only Paulis are diagonal between the two run-boundary
                    // Hadamard transforms, so no state permutation is needed.
                    let odd = ne_tile(
                        parity_large(andi(basis, broadcast_scalar(xmask, const_shape![1, 4096]))),
                        zero_basis,
                    );
                    let direction = select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                    let q = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 2i32),
                            const_shape![1, 4096],
                        );
                    re = c * old_re - q * old_im;
                    im = c * old_im + q * old_re;
                    if load_u64(metadata, meta + 9i32) != 0u64 {
                        let scale = broadcast_scalar(
                            load_f32(parameters, params + 3i32),
                            const_shape![1, 4096],
                        );
                        re = hadamard_mask_large(re, x_basis_mask) * scale;
                        im = hadamard_mask_large(im, x_basis_mask) * scale;
                    }
                } else {
                    let partner_re = flip_mask_large(re, xmask);
                    let partner_im = flip_mask_large(im, xmask);
                    if zmask == 0u64 {
                        let direction = select(sign_2d, negative_one_state, one_state);
                        let q = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 4096],
                            );
                        re = c * old_re - q * partner_im;
                        im = c * old_im + q * partner_re;
                    } else {
                        let partner_basis =
                            xori(basis, broadcast_scalar(xmask, const_shape![1, 4096]));
                        let odd = ne_tile(
                            parity_large(andi(
                                partner_basis,
                                broadcast_scalar(zmask, const_shape![1, 4096]),
                            )),
                            zero_basis,
                        );
                        let direction =
                            select(ne_tile(sign_2d, odd), negative_one_state, one_state);
                        let cr = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 1i32),
                                const_shape![1, 4096],
                            );
                        let ci = direction
                            * broadcast_scalar(
                                load_f32(parameters, params + 2i32),
                                const_shape![1, 4096],
                            );
                        re = c * old_re + cr * partner_re - ci * partner_im;
                        im = c * old_im + cr * partner_im + ci * partner_re;
                    }
                }
            } else if opcode == 1u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let sign_2d = sign
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 4096]);
                let c = broadcast_scalar(load_f32(parameters, params), const_shape![1, 4096]);
                let sin =
                    broadcast_scalar(load_f32(parameters, params + 1i32), const_shape![1, 4096]);
                // CPU promotion uses q=-sin for false and q=+sin for true.
                let q = select(sign_2d, sin, negative_one_state * sin);
                let physical_bit = load_u64(metadata, meta + 7i32);
                let copy_re = flip_dynamic_large(re, physical_bit);
                let copy_im = flip_dynamic_large(im, physical_bit);
                let physical_clear = eq_tile(
                    andi(
                        basis,
                        broadcast_scalar(bit_mask(physical_bit), const_shape![1, 4096]),
                    ),
                    zero_basis,
                );
                re = select(physical_clear, c * re, negative_one_state * q * copy_im);
                im = select(physical_clear, c * im, q * copy_re);
            } else if opcode == 2u64 {
                let xmask = load_u64(metadata, meta + 5i32);
                let zmask = load_u64(metadata, meta + 6i32);
                let pivot = load_u64(metadata, meta + 7i32);
                let diagonal_phase_word = load_u64(metadata, meta + 8i32);
                let diagonal_phase = diagonal_phase_word != 0u64;
                let z_without = load_u64(metadata, meta + 9i32);
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let diagonal = xmask == 0u64;
                let pivot_bit = bit_mask(pivot);
                let mut probability_true = zero_probability;
                let mut contribution_re = zero_state;
                let mut contribution_im = zero_state;
                if diagonal {
                    let odd = if zmask == 0u64 {
                        ne_tile(zero_basis, zero_basis)
                    } else {
                        ne_tile(
                            parity_large(andi(
                                basis,
                                broadcast_scalar(zmask, const_shape![1, 4096]),
                            )),
                            zero_basis,
                        )
                    };
                    let target = diagonal_phase_word == 0u64;
                    let selected = eq_tile(odd, broadcast_scalar(target, const_shape![1, 4096]));
                    let mass = select(selected, re * re + im * im, zero_state);
                    probability_true = reduce_sum(mass, 1i32);
                }
                if xmask != 0u64 {
                    let partner_re = flip_mask_large(re, xmask);
                    let partner_im = flip_mask_large(im, xmask);
                    let odd = if zmask == 0u64 {
                        ne_tile(zero_basis, zero_basis)
                    } else {
                        ne_tile(
                            parity_large(andi(
                                basis,
                                broadcast_scalar(zmask, const_shape![1, 4096]),
                            )),
                            zero_basis,
                        )
                    };
                    let direction = select(odd, one_state, negative_one_state);
                    let cr = direction
                        * broadcast_scalar(load_f32(parameters, params), const_shape![1, 4096]);
                    let ci = direction
                        * broadcast_scalar(
                            load_f32(parameters, params + 1i32),
                            const_shape![1, 4096],
                        );
                    contribution_re = cr * partner_re - ci * partner_im;
                    contribution_im = cr * partner_im + ci * partner_re;
                    let ar = inv_sqrt2_state * re + contribution_re;
                    let ai = inv_sqrt2_state * im + contribution_im;
                    let pivot_clear = eq_tile(
                        andi(basis, broadcast_scalar(pivot_bit, const_shape![1, 4096])),
                        zero_basis,
                    );
                    let mass = select(pivot_clear, ar * ar + ai * ai, zero_state);
                    probability_true = reduce_sum(mass, 1i32);
                }
                probability_true = minf(
                    maxf(
                        probability_true,
                        zero_probability,
                        nan::Enabled,
                        ftz::Disabled,
                    ),
                    one_probability,
                    nan::Enabled,
                    ftz::Disabled,
                );
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, probability_true);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }

                let branch_2d = branch
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![1, 4096]);
                let probability =
                    select(branch, probability_true, one_probability - probability_true);
                let invnorm = rsqrt(
                    maxf(probability, probability_floor, nan::Enabled, ftz::Disabled),
                    ftz::Disabled,
                )
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 4096]);
                let pivot_clear = eq_tile(
                    andi(basis, broadcast_scalar(pivot_bit, const_shape![1, 4096])),
                    zero_basis,
                );
                let mut out_re = zero_state;
                let mut out_im = zero_state;
                if diagonal {
                    let partner_re = flip_dynamic_large(re, pivot);
                    let partner_im = flip_dynamic_large(im, pivot);
                    let odd_bits = parity_large(andi(
                        basis,
                        broadcast_scalar(z_without, const_shape![1, 4096]),
                    ));
                    let odd = ne_tile(odd_bits, zero_basis);
                    let phase: Tile<bool, { [1, 4096] }> =
                        broadcast_scalar(diagonal_phase, const_shape![1, 4096]);
                    let odd_branch: Tile<bool, { [1, 4096] }> = ne_tile(odd, branch_2d);
                    let take_high = ne_tile(odd_branch, phase);
                    out_re = select(take_high, partner_re, re);
                    out_im = select(take_high, partner_im, im);
                }
                if xmask != 0u64 {
                    let signed_re = select(
                        branch_2d,
                        contribution_re,
                        negative_one_state * contribution_re,
                    );
                    let signed_im = select(
                        branch_2d,
                        contribution_im,
                        negative_one_state * contribution_im,
                    );
                    out_re = inv_sqrt2_state * re + signed_re;
                    out_im = inv_sqrt2_state * im + signed_im;
                }
                out_re = select(pivot_clear, out_re, zero_state);
                out_im = select(pivot_clear, out_im, zero_state);
                re = out_re * invnorm;
                im = out_im * invnorm;
            } else if opcode == 3u64 {
                let branch_slot = load_u64(metadata, meta + 10i32);
                let branch_bit = load_u64(metadata, meta + 11i32);
                let random_row = load_i32(controls, control);
                let uniform =
                    load_randoms(randoms, random_row, random_stride, shot_start, lanes, live);
                let branch = lt_tile(uniform, half_probability);
                let branch_bit = broadcast_scalar(branch_bit, const_shape![1]);
                let branch_value = select(branch, branch_bit, zero_shots);
                if branch_slot < 64u64 {
                    branches0 = ori(branches0, branch_value);
                } else if branch_slot < 128u64 {
                    branches1 = ori(branches1, branch_value);
                } else if branch_slot < 192u64 {
                    branches2 = ori(branches2, branch_value);
                } else {
                    branches3 = ori(branches3, branch_value);
                }
            } else if opcode == 5u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    select(sign, zero_probability - one_probability, one_probability),
                );
            } else if opcode == 6u64 {
                let sign = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                let probability_true = measurement_probability_large(
                    re,
                    im,
                    basis,
                    parameters,
                    params,
                    load_u64(metadata, meta + 5i32),
                    load_u64(metadata, meta + 6i32),
                    load_u64(metadata, meta + 7i32),
                    load_u64(metadata, meta + 8i32),
                );
                let expectation = select(sign, zero_probability - one_probability, one_probability)
                    * (one_probability
                        - broadcast_scalar(2.0f32, const_shape![1]) * probability_true);
                store_f32_row(
                    expectations,
                    load_i32(expectation_indices, instruction),
                    shots,
                    shot_start,
                    lanes,
                    live,
                    expectation,
                );
            } else {
                let outcome = instruction_expression_rows(
                    metadata,
                    controls,
                    expression_values,
                    random_stride,
                    shot_start,
                    lanes,
                    live,
                    instruction,
                    branches0,
                    branches1,
                    branches2,
                    branches3,
                );
                if keep_records != 0i32 {
                    store_u64_row(
                        detectors,
                        load_i32(expectation_indices, instruction),
                        shots,
                        shot_start,
                        lanes,
                        live,
                        select(outcome, one_shots, zero_shots),
                    );
                }
                if load_u64(metadata, meta + 8i32) != 0u64 {
                    discarded = ori(discarded, select(outcome, one_shots, zero_shots));
                    // A rejected one-shot tile cannot contribute a logical
                    // error, so its later quantum state is unobservable.
                    let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded, 0i32);
                    let discarded_count: u64 = tile_to_scalar(discarded_count);
                    if discarded_count != 0u64 {
                        break;
                    }
                }
            }
            instruction = instruction + 1i32;
        }

        let logical_values = load_u64_row(
            expression_values,
            logical_word,
            random_stride,
            shot_start,
            lanes,
            live,
        );
        let logical = expression(
            logical_values,
            logical_block_mask,
            logical_mask0,
            logical_mask1,
            logical_mask2,
            logical_mask3,
            branches0,
            branches1,
            branches2,
            branches3,
        );
        let live_bits = select(live, one_shots, zero_shots);
        let discarded_bits = andi(discarded, live_bits);
        let logical_bits = select(logical, one_shots, zero_shots);
        let accepted = eq_tile(discarded_bits, zero_shots);
        let logical_error_bits = andi(select(accepted, logical_bits, zero_shots), live_bits);
        let discarded_count: Tile<u64, { [] }> = reduce_sum(discarded_bits, 0i32);
        let logical_error_count: Tile<u64, { [] }> = reduce_sum(logical_error_bits, 0i32);
        store_u64(block_counts, pid * 2i32, discarded_count);
        store_u64(block_counts, pid * 2i32 + 1i32, logical_error_count);
    }
}

pub use kernels::{
    apply_sparse_exogenous, evaluate_exogenous_partials, reduce_block_counts,
    reduce_exogenous_partials, sample16, sample128, sample1024, sample4096, wide_sample_finalize,
    wide_sample_init, wide_sample_step,
};
