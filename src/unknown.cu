#include "unknown.cuh"

// ########################################## EXECUTION FUNCTIONS ##########################################

/// @brief FEEDS A CORTEX THROUGH THE PROVIDED INPUT2D. INPUT DATA SHOULD ALREADY BE IN THE PROVIDED INPUT2D
/// WHEN THIS FUNCTION IS CALLED.
/// @param cortex THE CORTEX TO FEED THE INPUT INTO
/// @param input THE INPUT STRUCTURE CONTAINING THE DATA TO FEED
__global__ void c2d_feed2d(unk_cortex2d_t *cortex, unk_input2d_t *input)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= input->x1 - input->x0 || y >= input->y1 - input->y0)
    {
        return;
    }
    unk_bool_t excite = value_to_pulse(cortex->sample_window, cortex->ticks_count % cortex->sample_window, input->values[IDX2D(x, y, input->x1 - input->x0)], cortex->pulse_mapping);
    if (excite)
    {
        cortex->neurons[IDX2D(x + input->x0, y + input->y0, cortex->width)].value += input->exc_value;
    }
}

/// @brief READS DATA FROM A CORTEX INTO THE PROVIDED OUTPUT2D STRUCTURE. OUTPUT DATA WILL BE
/// STORED IN THE PROVIDED OUTPUT2D AFTER COMPLETION.
/// @param cortex THE CORTEX TO READ VALUES FROM
/// @param output THE OUTPUT STRUCTURE TO STORE THE READ DATA
__global__ void c2d_read2d(unk_cortex2d_t *cortex, unk_output2d_t *output)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= output->x1 - output->x0 || y >= output->y1 - output->y0)
    {
        return;
    }
    // [TODO]
}

/// @brief PERFORMS A FULL RUN CYCLE OVER THE PROVIDED CORTEX. THIS UPDATES THE CORTEX STATE
/// BASED ON CURRENT INPUTS AND INTERNAL STATE.
/// @param prev_cortex THE CORTEX AT ITS CURRENT STATE
/// @param next_cortex THE CORTEX THAT WILL BE UPDATED BY THE TICK CYCLE
/// @warning PREV_CORTEX AND NEXT_CORTEX MUST BE IDENTICAL COPIES, OTHERWISE THE OPERATION MAY FAIL
__global__ void c2d_tick(unk_cortex2d_t *prev_cortex, unk_cortex2d_t *next_cortex)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= prev_cortex->width || y >= prev_cortex->height)
    {
        return;
    }
    unk_cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);
    unk_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
    unk_neuron_t *next_neuron = &(next_cortex->neurons[neuron_index]);
    *next_neuron = prev_neuron;
    unk_cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);
    unk_nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
    unk_nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
    unk_nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
    unk_nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
    unk_nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;
    bool evolve = (prev_cortex->ticks_count % (((unk_evol_step_t)prev_cortex->evol_step) + 1)) == 0;
    for (unk_nh_radius_t j = 0; j < nh_diameter; j++)
    {
        for (unk_nh_radius_t i = 0; i < nh_diameter; i++)
        {
            unk_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
            unk_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);
            if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height))
            {
                unk_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                unk_cortex_size_t neighbor_index = IDX2D(WRAP(neighbor_x, prev_cortex->width),
                                                         WRAP(neighbor_y, prev_cortex->height),
                                                         prev_cortex->width);
                unk_neuron_t neighbor = prev_cortex->neurons[neighbor_index];
                unk_syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) |
                                                  ((prev_str_mask_b & 0x01U) << 0x01U) |
                                                  ((prev_str_mask_c & 0x01U) << 0x02U);
                next_neuron->rand_state = xorshf32(next_neuron->rand_state);
                unk_chance_t random = next_neuron->rand_state % 0xFFFFU;
                unk_syn_strength_t strength_diff = UNK_MAX_SYN_STRENGTH - syn_strength;
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
                if (evolve)
                {
                    if (!(prev_ac_mask & 0x01U) &&
                        prev_neuron.syn_count < next_neuron->max_syn_count &&
                        // [TODO] Make sure there's no overflow.
                        random < prev_cortex->syngen_chance * (unk_chance_t)neighbor.pulse)
                    {
                        next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);
                        if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio)
                        {
                            next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                        }
                        else
                        {
                            next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                        }
                        next_neuron->syn_count++;
                    }
                    else if (prev_ac_mask & 0x01U &&
                             syn_strength <= 0x00U &&
                             random < prev_cortex->syngen_chance / (neighbor.pulse + 1))
                    {
                        next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->syn_count--;
                    }
                    if (prev_ac_mask & 0x01U)
                    {
                        if (syn_strength < UNK_MAX_SYN_STRENGTH &&
                            prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                            // [TODO] Make sure there's no overflow.
                            random < prev_cortex->synstr_chance * (unk_chance_t)neighbor.pulse * (unk_chance_t)strength_diff)
                        {
                            syn_strength++;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                            next_neuron->tot_syn_strength++;
                        }
                        else if (syn_strength > 0x00U &&
                                 // [TODO] Make sure there's no overflow.
                                 random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1))
                        {
                            syn_strength--;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                            next_neuron->tot_syn_strength--;
                        }
                    }
                    next_cortex->evols_count++;
                }
            }
            prev_ac_mask >>= 0x01U;
            prev_exc_mask >>= 0x01U;
            prev_str_mask_a >>= 0x01U;
            prev_str_mask_b >>= 0x01U;
            prev_str_mask_c >>= 0x01U;
        }
    }
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
        next_neuron->pulse--;
    }
    next_neuron->pulse_mask <<= 0x01U;
    if (prev_neuron.value > prev_cortex->fire_threshold + prev_neuron.pulse)
    {
        next_neuron->value = next_cortex->recovery_value;
        next_neuron->pulse_mask |= 0x01U;
        next_neuron->pulse++;
    }
    next_cortex->ticks_count++;
}

// ########################################## INPUT MAPPING FUNCTIONS ##########################################

/// @brief CALCULATES THE PULSE PATTERN FOR THE DIFFERENTIAL FAST PROPORTIONAL MAPPING ALGORITHM
/// @param step THE CURRENT STEP POSITION IN THE WINDOW
/// @param input THE INPUT VALUE TO MAP
/// @param upper THE UPPER BOUND OF THE SAMPLING WINDOW
/// @param rounded TRUE IF THE DIVISION SHOULD BE ROUNDED, FALSE OTHERWISE
/// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
__host__ __device__ static inline unk_bool_t calc_prop_pulse(unk_ticks_count_t step, unk_ticks_count_t input,
                                                             unk_ticks_count_t upper, unk_bool_t rounded)
{
    if (input < upper / 2)
    {
        if (input == 0) return UNK_TRUE;
        unk_ticks_count_t div = rounded ? (unk_ticks_count_t)round((double)upper / (double)input) : upper / input;
        return step % div == 0;
    }
    else
    {
        if (input >= upper) return UNK_TRUE;
        unk_ticks_count_t div = rounded ? (unk_ticks_count_t)round((double)upper / (double)(upper - input)) : upper / (upper - input);
        return step % div != 0;
    }
}

/// @brief MAPS AN INPUT VALUE TO A PULSE PATTERN USING THE SPECIFIED MAPPING ALGORITHM
/// ALL MAPPING ALGORITHMS ARE IMPLEMENTED WITHIN THIS SINGLE FUNCTION:
/// - LINEAR: SIMPLE UNIFORM DISTRIBUTION WITH MINIMUM ONE PULSE;
/// - FPROP: FAST PROPORTIONAL FOR SMALL WINDOWS;
/// - RPROP: PRECISE PROPORTIONAL FOR LARGE WINDOWS;
/// - DFPROP: DIFFERENTIAL FAST PROPORTIONAL FOR RATE-OF-CHANGE DETECTION (SUITABLE FOR APPLICATIONS REQUIRING SENSITIVITY TO INPUT VARIATION).
/// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
/// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
/// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..(SAMPLE_WINDOW - 1))
/// @param pulse_mapping THE MAPPING ALGORITHM TO USE FOR PULSE GENERATION
/// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
__host__ __device__ unk_bool_t value_to_pulse(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                              unk_ticks_count_t input, unk_pulse_mapping_t pulse_mapping)
{
    if (input >= sample_window) return UNK_FALSE;
    const unk_ticks_count_t upper = sample_window - 1;
    switch (pulse_mapping)
    {
        case UNK_PULSE_MAPPING_LINEAR:
            return input < sample_window && sample_step % (sample_window - input) == 0;
        case UNK_PULSE_MAPPING_FPROP:
            return calc_prop_pulse(sample_step, input, upper, UNK_FALSE);
        case UNK_PULSE_MAPPING_RPROP:
            return calc_prop_pulse(sample_step, input, upper, UNK_TRUE);
        case UNK_PULSE_MAPPING_DFPROP:
        {
            static unk_ticks_count_t prev_input = 0;
            static unk_ticks_count_t prev_sample_window = 0;
            if (prev_sample_window != sample_window)
            {
                prev_input = 0;
                prev_sample_window = sample_window;
            }
            unk_ticks_count_t input_diff = (input > prev_input) ? input - prev_input : prev_input - input;
            prev_input = input;
            if (input_diff > 0)
            {
                unk_ticks_count_t pulse_rate = (sample_window - input_diff) / 2;
                return pulse_rate == 0 || sample_step % (pulse_rate + 1) == 0;
            }
            return calc_prop_pulse(sample_step, input, upper, UNK_FALSE);
        }
    }
    return UNK_FALSE;
}