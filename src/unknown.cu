#include "unknown.h"

// ########################################## EXECUTION FUNCTIONS ##########################################

__global__ void c2d_feed2d(unk_cortex2d_t *cortex, unk_input2d_t *input)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= input->x1 - input->x0 || y >= input->y1 - input->y0)
    {
        return;
    }

    // Check whether the current input neuron should be excited or not.
    unk_bool_t excite = value_to_pulse(
        cortex->sample_window,
        cortex->ticks_count % cortex->sample_window,
        input->values[IDX2D(
            x,
            y,
            input->x1 - input->x0)],
        cortex->pulse_mapping);

    if (excite)
    {
        cortex->neurons[IDX2D(x + input->x0, y + input->y0, cortex->width)].value += input->exc_value;
    }
}

__global__ void c2d_read2d(unk_cortex2d_t *cortex, unk_output2d_t *output)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= output->x1 - output->x0 || y >= output->y1 - output->y0)
    {
        return;
    }

    // TODO.
}

__global__ void c2d_tick(unk_cortex2d_t *prev_cortex, unk_cortex2d_t *next_cortex)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= prev_cortex->width || y >= prev_cortex->height)
    {
        return;
    }

    // Retrieve the involved neurons.
    unk_cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);
    unk_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
    unk_neuron_t *next_neuron = &(next_cortex->neurons[neuron_index]);

    // Copy prev neuron values to the new one.
    *next_neuron = prev_neuron;

    /* Compute the neighborhood diameter:
        d = 7
        <------------->
        r = 3
        <----->
        +-|-|-|-|-|-|-+
        |             |
        |             |
        |      X      |
        |             |
        |             |
        +-|-|-|-|-|-|-+
    */
    unk_cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);

    unk_nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
    unk_nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
    unk_nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
    unk_nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
    unk_nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;

    // Defines whether to evolve or not.
    // evol_step is incremented by 1 to account for edge cases and human readable behavior:
    // 0x0000 -> 0 + 1 = 1, so the cortex evolves at every tick, meaning that there are no free ticks between evolutions.
    // 0xFFFF -> 65535 + 1 = 65536, so the cortex never evolves, meaning that there is an infinite amount of ticks between evolutions.
    bool evolve = (prev_cortex->ticks_count % (((unk_evol_step_t)prev_cortex->evol_step) + 1)) == 0;

    // Increment the current neuron value by reading its connected neighbors.
    for (unk_nh_radius_t j = 0; j < nh_diameter; j++)
    {
        for (unk_nh_radius_t i = 0; i < nh_diameter; i++)
        {
            unk_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
            unk_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);

            // Exclude the central neuron from the list of neighbors.
            if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height))
            {
                // The index of the current neighbor in the current neuron's neighborhood.
                unk_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                unk_cortex_size_t neighbor_index = IDX2D(WRAP(neighbor_x, prev_cortex->width),
                                                         WRAP(neighbor_y, prev_cortex->height),
                                                         prev_cortex->width);

                // Fetch the current neighbor.
                unk_neuron_t neighbor = prev_cortex->neurons[neighbor_index];

                // Compute the current synapse strength.
                unk_syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) |
                                                  ((prev_str_mask_b & 0x01U) << 0x01U) |
                                                  ((prev_str_mask_c & 0x01U) << 0x02U);

                // Pick a random number for each neighbor, capped to the max uint16 value.
                next_neuron->rand_state = xorshf32(next_neuron->rand_state);
                unk_chance_t random = next_neuron->rand_state % 0xFFFFU;

                // Inverse of the current synapse strength, useful when computing depression probability (synapse deletion and weakening).
                unk_syn_strength_t strength_diff = UNK_MAX_SYN_STRENGTH - syn_strength;

                // Check if the last bit of the mask is 1 or 0: 1 = active synapse, 0 = inactive synapse.
                if (prev_ac_mask & 0x01U)
                {
                    unk_neuron_value_t neighbor_influence = (prev_exc_mask & 0x01U ? prev_cortex->exc_value : -prev_cortex->exc_value) * ((syn_strength / 4) + 1);
                    if (neighbor.value > prev_cortex->fire_threshold)
                    {
                        if (next_neuron->value + neighbor_influence < prev_cortex->recovery_value)
                        {
                            next_neuron->value = prev_cortex->recovery_value;
                        }
                        else
                        {
                            next_neuron->value += neighbor_influence;
                        }
                    }
                }

                // Perform the evolution phase if allowed.
                if (evolve)
                {
                    // Structural plasticity: create or destroy a synapse.
                    if (!(prev_ac_mask & 0x01U) &&
                        prev_neuron.syn_count < next_neuron->max_syn_count &&
                        // Frequency component.
                        // TODO Make sure there's no overflow.
                        random < prev_cortex->syngen_chance * (unk_chance_t)neighbor.pulse)
                    {
                        // Add synapse.
                        next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);

                        // Set the new synapse's strength to 0.
                        next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);

                        // Define whether the new synapse is excitatory or inhibitory.
                        if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio)
                        {
                            // Inhibitory.
                            next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                        }
                        else
                        {
                            // Excitatory.
                            next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                        }

                        next_neuron->syn_count++;
                    }
                    else if (prev_ac_mask & 0x01U &&
                             // Only 0-strength synapses can be deleted.
                             syn_strength <= 0x00U &&
                             // Frequency component.
                             random < prev_cortex->syngen_chance / (neighbor.pulse + 1))
                    {
                        // Delete synapse.
                        next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);

                        next_neuron->syn_count--;
                    }

                    // Functional plasticity: strengthen or weaken a synapse.
                    if (prev_ac_mask & 0x01U)
                    {
                        if (syn_strength < UNK_MAX_SYN_STRENGTH &&
                            prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                            // TODO Make sure there's no overflow.
                            random < prev_cortex->synstr_chance * (unk_chance_t)neighbor.pulse * (unk_chance_t)strength_diff)
                        {
                            syn_strength++;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                            next_neuron->tot_syn_strength++;
                        }
                        else if (syn_strength > 0x00U &&
                                 // TODO Make sure there's no overflow.
                                 random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1))
                        {
                            syn_strength--;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                            next_neuron->tot_syn_strength--;
                        }
                    }

                    // Increment evolutions count.
                    next_cortex->evols_count++;
                }
            }

            // Shift the masks to check for the next neighbor.
            prev_ac_mask >>= 0x01U;
            prev_exc_mask >>= 0x01U;
            prev_str_mask_a >>= 0x01U;
            prev_str_mask_b >>= 0x01U;
            prev_str_mask_c >>= 0x01U;
        }
    }

    // Push to equilibrium by decaying to zero, both from above and below.
    if (prev_neuron.value > 0x00)
    {
        next_neuron->value -= next_cortex->decay_value;
    }
    else if (prev_neuron.value < 0x00)
    {
        next_neuron->value += next_cortex->decay_value;
    }

    if ((prev_neuron.pulse_mask >> prev_cortex->pulse_window) & 0x01U)
    {
        // Decrease pulse if the oldest recorded pulse is active.
        next_neuron->pulse--;
    }

    next_neuron->pulse_mask <<= 0x01U;

    // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
    if (prev_neuron.value > prev_cortex->fire_threshold + prev_neuron.pulse)
    {
        // Fired at the previous step.
        next_neuron->value = next_cortex->recovery_value;

        // Store pulse.
        next_neuron->pulse_mask |= 0x01U;
        next_neuron->pulse++;
    }

    next_cortex->ticks_count++;
}

// ########################################## INPUT MAPPING FUNCTIONS ##########################################

__host__ __device__ unk_bool_t value_to_pulse(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step, unk_ticks_count_t input, unk_pulse_mapping_t pulse_mapping)
{
    if (input < sample_window)
    {
        unk_ticks_count_t upper = sample_window - 1;
        switch (pulse_mapping)
        {
        case UNK_PULSE_MAPPING_LINEAR: ;
            return (sample_step % (sample_window - input) == 0) ? UNK_TRUE : UNK_FALSE;
        case UNK_PULSE_MAPPING_FPROP: ;
            if (input < sample_window / 2)
            {
                if ((sample_step == 0) ||
                    (input > 0 && sample_step % (upper / input) == 0))
                {
                    return UNK_TRUE;
                }
            }
            else
            {
                if (input >= upper || sample_step % (upper / (upper - input)) != 0)
                {
                    return UNK_TRUE;
                }
            }
            return UNK_FALSE;
        case UNK_PULSE_MAPPING_RPROP: ;
            if (input < sample_window / 2)
            {
                if ((sample_step == 0) ||
                    (input > 0 && sample_step % (upper / input) == 0))
                {
                    return UNK_TRUE;
                }
            }
            else
            {
                if (input >= upper || sample_step % (upper / (upper - input)) != 0)
                {
                    return UNK_TRUE;
                }
            }
            return UNK_FALSE;
        case UNK_PULSE_MAPPING_DFPROP: ;
            if (input < sample_window / 2)
            {
                if ((sample_step == 0) || (input > 0 && sample_step % (upper / (input * 2)) == 0))
                {
                    return UNK_TRUE;
                }
            }
            else
            {
                if (input >= upper || sample_step % (upper / ((upper - input) * 2)) != 0)
                {
                    return UNK_TRUE;
                }
            }
            return UNK_FALSE;
        }
    }
    return UNK_FALSE;
}