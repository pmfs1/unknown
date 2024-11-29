#include "unknown.h"

void c2d_feed2d(unk_cortex2d_t *cortex, unk_input2d_t *input)
{
#pragma omp parallel for collapse(2)
    for (unk_cortex_size_t y = input->y0; y < input->y1; y++)
    {
        for (unk_cortex_size_t x = input->x0; x < input->x1; x++)
        {
            unk_bool_t excite = value_to_pulse(
                cortex->sample_window,
                cortex->ticks_count % cortex->sample_window,
                input->values[IDX2D(
                    x - input->x0,
                    y - input->y0,
                    input->x1 - input->x0)],
                cortex->pulse_mapping);
            if (excite)
            {
                cortex->neurons[IDX2D(x, y, cortex->width)].value += input->exc_value;
            }
        }
    }
}

void c2d_read2d(unk_cortex2d_t *cortex, unk_output2d_t *output)
{
#pragma omp parallel for collapse(2)
    for (unk_cortex_size_t y = output->y0; y < output->y1; y++)
    {
        for (unk_cortex_size_t x = output->x0; x < output->x1; x++)
        {
            output->values[IDX2D(
                x - output->x0,
                y - output->y0,
                output->x1 - output->x0)] = cortex->neurons[IDX2D(x,
                                                                  y,
                                                                  cortex->width)]
                                                .pulse;
        }
    }
}

void c2d_tick(unk_cortex2d_t *prev_cortex, unk_cortex2d_t *next_cortex)
{
#pragma omp parallel for collapse(2)
    for (unk_cortex_size_t y = 0; y < prev_cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < prev_cortex->width; x++)
        {
            // RETRIEVE THE NEURONS FROM PREVIOUS AND NEXT CORTEX STATES
            unk_cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);
            unk_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
            unk_neuron_t *next_neuron = &(next_cortex->neurons[neuron_index]);
            // COPY PREVIOUS NEURON STATE TO THE NEW ONE
            *next_neuron = prev_neuron;
            /* NEIGHBORHOOD CONFIGURATION AND NEURAL CONNECTIVITY:
             * IMPLEMENTS A SQUARE GRID TOPOLOGY WITH:
             * - CONFIGURABLE RADIUS FOR FLEXIBLE CONNECTIVITY PATTERNS
             * - SYMMETRIC CONNECTIVITY FOR UNIFORM SIGNAL PROPAGATION
             * - BOUNDARY HANDLING FOR EDGE CASES
             *
             * ARCHITECTURE VISUALIZATION:
             *    TOTAL DIAMETER (d) = 2r + 1
             * <----------------------->
             *        RADIUS (r)
             *      <----------->
             *    +-|-|-|-|-|-|-|-|+
             *    |               | |
             *    |    ACTIVE    | |
             *    |     ZONE     | |
             *    |      X       | | <- CENTER NEURON (CURRENT)
             *    |    RADIUS    | |
             *    |               | |
             *    +-|-|-|-|-|-|-|-|+
             *
             * WHERE:
             * - X: CURRENT NEURON BEING PROCESSED
             * - ACTIVE ZONE: REGION OF POTENTIAL SYNAPTIC CONNECTIONS
             * - RADIUS: MAXIMUM DISTANCE FOR SYNAPTIC CONNECTIONS
             */
            unk_cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);
            unk_nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
            unk_nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
            unk_nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
            unk_nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
            unk_nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;
            // DETERMINE IF EVOLUTION SHOULD OCCUR THIS TICK
            // EVOLUTION STEP MECHANICS:
            // - ADD 1 TO HANDLE EDGE CASES AND ENSURE HUMAN-READABLE BEHAVIOR
            // - 0x0000 -> 1 TICK BETWEEN EVOLUTIONS (EVOLVES EVERY TICK)
            // - 0xFFFF -> 65536 TICKS BETWEEN EVOLUTIONS (NEVER EVOLVES)
            unk_bool_t evolve = (prev_cortex->ticks_count % (((unk_evol_step_t)prev_cortex->evol_step) + 1)) == 0;
            // PROCESS EACH NEIGHBOR IN THE DEFINED NEIGHBORHOOD
            for (unk_nh_radius_t j = 0; j < nh_diameter; j++)
            {
                for (unk_nh_radius_t i = 0; i < nh_diameter; i++)
                {
                    unk_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
                    unk_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);
                    // SKIP CENTER NEURON AND CHECK BOUNDARY CONDITIONS
                    if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                        (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height))
                    {
                        // THE INDEX OF THE CURRENT NEIGHBOR IN THE CURRENT NEURON'S NEIGHBORHOOD
                        unk_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                        unk_cortex_size_t neighbor_index = IDX2D(WRAP(neighbor_x, prev_cortex->width),
                                                                 WRAP(neighbor_y, prev_cortex->height),
                                                                 prev_cortex->width);
                        // FETCH THE CURRENT NEIGHBOR
                        unk_neuron_t neighbor = prev_cortex->neurons[neighbor_index];
                        // COMPUTE THE CURRENT SYNAPSE STRENGTH
                        unk_syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) |
                                                          ((prev_str_mask_b & 0x01U) << 0x01U) |
                                                          ((prev_str_mask_c & 0x01U) << 0x02U);
                        // PICK A RANDOM NUMBER FOR EACH NEIGHBOR, CAPPED TO THE MAX UINT16 VALUE
                        next_neuron->rand_state = xorshf32(next_neuron->rand_state);
                        unk_chance_t random = next_neuron->rand_state % 0xFFFFU;
                        // INVERSE OF THE CURRENT SYNAPSE STRENGTH, USEFUL WHEN COMPUTING DEPRESSION PROBABILITY (SYNAPSE DELETION AND WEAKENING)
                        unk_syn_strength_t strength_diff = UNK_MAX_SYN_STRENGTH - syn_strength;
                        // CHECK IF THE LAST BIT OF THE MASK IS 1 OR 0: 1 = ACTIVE SYNAPSE, 0 = INACTIVE SYNAPSE
                        if (prev_ac_mask & 0x01U)
                        {
                            unk_neuron_value_t neighbor_influence = ((prev_exc_mask & 0x01U)
                                                                         ? prev_cortex->exc_value
                                                                         : -prev_cortex->exc_value) *
                                                                    ((syn_strength /
                                                                      4) +
                                                                     1);
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
                        /* SYNAPTIC PLASTICITY IMPLEMENTATION:
                         *
                         * 1. STRUCTURAL PLASTICITY:
                         *    - SYNAPSE CREATION:
                         *      * ACTIVITY-DEPENDENT FORMATION
                         *      * PROBABILITY-BASED GENERATION
                         *      * CAPACITY-LIMITED GROWTH
                         *    - SYNAPSE PRUNING:
                         *      * WEAKNESS-BASED REMOVAL
                         *      * ACTIVITY-DEPENDENT SURVIVAL
                         *      * HOMEOSTATIC REGULATION
                         *
                         * 2. FUNCTIONAL PLASTICITY:
                         *    - STRENGTH MODULATION:
                         *      * DISCRETE STRENGTH LEVELS
                         *      * BOUNDED WEIGHT CHANGES
                         *      * ACTIVITY-DEPENDENT SCALING
                         *    - HOMEOSTATIC CONTROL:
                         *      * TOTAL STRENGTH LIMITS
                         *      * BALANCE MAINTENANCE
                         *      * STABILITY PRESERVATION
                         */
                        // STRUCTURAL PLASTICITY: SYNAPSE CREATION AND DESTRUCTION
                        if (evolve)
                        {
                            // CREATE NEW SYNAPSE IF CONDITIONS ARE MET:
                            // 1. NO EXISTING SYNAPSE
                            // 2. BELOW MAX SYNAPSE COUNT
                            // 3. PASSES RANDOM PROBABILITY CHECK
                            // FREQUENCY COMPONENT
                            if (!(prev_ac_mask & 0x01U) &&
                                prev_neuron.syn_count < next_neuron->max_syn_count &&
                                random < prev_cortex->syngen_chance * (unk_chance_t)neighbor.pulse)
                            {
                                // ADD SYNAPSE
                                next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);

                                // SET THE NEW SYNAPSE'S STRENGTH TO 0
                                next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                                next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                                next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);

                                // DEFINE WHETHER THE NEW SYNAPSE IS EXCITATORY OR INHIBITORY
                                if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio)
                                {
                                    // INHIBITORY
                                    next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                                }
                                else
                                {
                                    // EXCITATORY
                                    next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                                }

                                next_neuron->syn_count++;
                            }
                            // DELETE EXISTING SYNAPSE IF:
                            // 1. SYNAPSE EXISTS
                            // 2. STRENGTH IS ZERO
                            // 3. PASSES PROBABILITY CHECK BASED ON NEIGHBOR ACTIVITY
                            else if (prev_ac_mask & 0x01U &&
                                     // ONLY 0-STRENGTH SYNAPSES CAN BE DELETED
                                     syn_strength == 0x00U &&
                                     // FREQUENCY COMPONENT
                                     random < prev_cortex->syngen_chance / (neighbor.pulse + 1))
                            {
                                // DELETE SYNAPSE
                                next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);

                                next_neuron->syn_count--;
                            }
                            // FUNCTIONAL PLASTICITY: MODIFY SYNAPSE STRENGTH
                            if (prev_ac_mask & 0x01U)
                            {
                                // STRENGTHEN SYNAPSE IF:
                                // 1. NOT AT MAX STRENGTH
                                // 2. TOTAL STRENGTH BELOW MAXIMUM
                                // 3. PASSES PROBABILITY CHECK
                                if (syn_strength < UNK_MAX_SYN_STRENGTH &&
                                    prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                                    random < prev_cortex->synstr_chance * (unk_chance_t)neighbor.pulse * (unk_chance_t)strength_diff)
                                {
                                    syn_strength++;
                                    next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->tot_syn_strength++;
                                }
                                // WEAKEN SYNAPSE IF PASSES PROBABILITY CHECK
                                else if (syn_strength > 0x00U &&
                                         random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1))
                                {
                                    syn_strength--;
                                    next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->tot_syn_strength--;
                                }
                            }
                            // INCREMENT EVOLUTIONS COUNT
                            next_cortex->evols_count++;
                        }
                    }
                    // SHIFT THE MASKS TO CHECK FOR THE NEXT NEIGHBOR
                    prev_ac_mask >>= 0x01U;
                    prev_exc_mask >>= 0x01U;
                    prev_str_mask_a >>= 0x01U;
                    prev_str_mask_b >>= 0x01U;
                    prev_str_mask_c >>= 0x01U;
                }
            }
            /* PULSE PROCESSING AND NEURAL DYNAMICS:
             *
             * 1. DECAY PHASE:
             *    - LEAKY INTEGRATE-AND-FIRE MODEL
             *    - BIDIRECTIONAL VALUE DECAY
             *    - EQUILIBRIUM TARGETING
             *
             * 2. PULSE MANAGEMENT:
             *    - TEMPORAL PULSE TRACKING
             *    - SLIDING WINDOW HISTORY
             *    - FREQUENCY MONITORING
             *
             * 3. FIRING MECHANICS:
             *    - ADAPTIVE THRESHOLDING
             *    - REFRACTORY ENFORCEMENT
             *    - RECOVERY REGULATION
             */
            // APPLY DECAY TO PUSH NEURON VALUE TOWARDS EQUILIBRIUM (ZERO)
            if (prev_neuron.value > 0x00)
            {
                next_neuron->value -= next_cortex->decay_value;
            }
            else if (prev_neuron.value < 0x00)
            {
                next_neuron->value += next_cortex->decay_value;
            }
            // UPDATE PULSE HISTORY AND HANDLE FIRING
            if ((prev_neuron.pulse_mask >> prev_cortex->pulse_window) & 0x01U)
            {
                // DECREASE PULSE COUNT IF OLDEST RECORDED PULSE IS ACTIVE
                next_neuron->pulse--;
            }
            // SHIFT PULSE HISTORY
            next_neuron->pulse_mask <<= 0x01U;
            // HANDLE NEURON FIRING AND RECOVERY:
            // - IF ABOVE THRESHOLD + CURRENT PULSE COUNT: FIRE AND ENTER RECOVERY
            // - STORE NEW PULSE IN HISTORY AND INCREMENT PULSE COUNT
            if (prev_neuron.value > prev_cortex->fire_threshold + prev_neuron.pulse)
            {
                // FIRED AT THE PREVIOUS STEP
                next_neuron->value = next_cortex->recovery_value;
                // STORE PULSE
                next_neuron->pulse_mask |= 0x01U;
                next_neuron->pulse++;
            }
        }
    }
    next_cortex->ticks_count++;
}

unk_bool_t value_to_pulse(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step, unk_ticks_count_t input,
                          unk_pulse_mapping_t pulse_mapping)
{
    if (input < sample_window)
    {
        unk_ticks_count_t upper = sample_window - 1;
        switch (pulse_mapping)
        {
        case UNK_PULSE_MAPPING_LINEAR:;
            return sample_step % (sample_window - input) == 0;
        case UNK_PULSE_MAPPING_FPROP:;
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
        case UNK_PULSE_MAPPING_RPROP:;
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
        case UNK_PULSE_MAPPING_DFPROP:;
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