#ifndef __UNKNOWN__
#define __UNKNOWN__

#include <stdint.h>
#include <stdlib.h>
// [TODO] Remove in release.
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "cortex.h"
#include "population.h"
#include "error.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // ########################################## EXECUTION FUNCTIONS ##########################################

    /// @brief FEEDS A CORTEX THROUGH THE PROVIDED INPUT2D. INPUT DATA SHOULD ALREADY BE IN THE PROVIDED INPUT2D
    /// WHEN THIS FUNCTION IS CALLED.
    /// @param cortex THE CORTEX TO FEED THE INPUT INTO
    /// @param input THE INPUT STRUCTURE CONTAINING THE DATA TO FEED
    void c2d_feed2d(unk_cortex2d_t *cortex, unk_input2d_t *input);

    /// @brief READS DATA FROM A CORTEX INTO THE PROVIDED OUTPUT2D STRUCTURE. OUTPUT DATA WILL BE
    /// STORED IN THE PROVIDED OUTPUT2D AFTER COMPLETION.
    /// @param cortex THE CORTEX TO READ VALUES FROM
    /// @param output THE OUTPUT STRUCTURE TO STORE THE READ DATA
    void c2d_read2d(unk_cortex2d_t *cortex, unk_output2d_t *output);

    /// @brief PERFORMS A FULL RUN CYCLE OVER THE PROVIDED CORTEX. THIS UPDATES THE CORTEX STATE
    /// BASED ON CURRENT INPUTS AND INTERNAL STATE.
    /// @param prev_cortex THE CORTEX AT ITS CURRENT STATE
    /// @param next_cortex THE CORTEX THAT WILL BE UPDATED BY THE TICK CYCLE
    /// @warning PREV_CORTEX AND NEXT_CORTEX MUST BE IDENTICAL COPIES, OTHERWISE THE OPERATION MAY FAIL
    void c2d_tick(unk_cortex2d_t *prev_cortex, unk_cortex2d_t *next_cortex);

    // ########################################## INPUT MAPPING FUNCTIONS ##########################################

    /// @brief MAPS AN INPUT VALUE TO A PULSE PATTERN USING THE SPECIFIED MAPPING ALGORITHM
    /// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
    /// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
    /// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..(SAMPLE_WINDOW - 1))
    /// @param pulse_mapping THE MAPPING ALGORITHM TO USE FOR PULSE GENERATION
    /// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
    unk_bool_t value_to_pulse(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step, unk_ticks_count_t input,
                              unk_pulse_mapping_t pulse_mapping);

    /// @brief COMPUTES A LINEAR PULSE MAPPING. GUARANTEES AT LEAST ONE PULSE EVEN FOR ZERO INPUT.
    /// PROVIDES THE SIMPLEST DISTRIBUTION OF PULSES.
    /// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
    /// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
    /// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..SAMPLE_WINDOW)
    /// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
    unk_bool_t value_to_pulse_linear(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                     unk_ticks_count_t input);

    /// @brief COMPUTES A FAST PROPORTIONAL MAPPING. OPTIMIZED FOR PERFORMANCE WITH SMALLER WINDOWS.
    /// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
    /// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
    /// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..SAMPLE_WINDOW)
    /// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
    unk_bool_t value_to_pulse_fprop(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                    unk_ticks_count_t input);

    /// @brief COMPUTES A PRECISE PROPORTIONAL MAPPING WITH BETTER PULSE DISTRIBUTION.
    /// RECOMMENDED FOR LARGER WINDOWS WHERE PRECISE DISTRIBUTION IS CRITICAL.
    /// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
    /// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
    /// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..SAMPLE_WINDOW)
    /// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
    unk_bool_t value_to_pulse_rprop(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                    unk_ticks_count_t input);

    /// @brief COMPUTES A DIFFERENTIAL FAST PROPORTIONAL MAPPING. OPTIMIZED FOR RATE-OF-CHANGE DETECTION.
    /// SUITABLE FOR APPLICATIONS REQUIRING SENSITIVITY TO INPUT VARIATIONS.
    /// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
    /// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
    /// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..SAMPLE_WINDOW)
    /// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
    unk_bool_t value_to_pulse_dfprop(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                     unk_ticks_count_t input);

#ifdef __cplusplus
}
#endif

#endif // __UNKNOWN__