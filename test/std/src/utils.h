#ifndef __UTILS__
#define __UTILS__

#define _POSIX_C_SOURCE 199309L // FOR CLOCK_GETTIME

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unknown/unknown.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// TIME CONVERSION MACROS
/// CONVERT SECONDS TO MILLISECONDS
#define S_TO_MS(s) ((s) * 1e3)
/// CONVERT SECONDS TO MICROSECONDS
#define S_TO_US(s) ((s) * 1e6)
/// CONVERT SECONDS TO NANOSECONDS
#define S_TO_NS(s) ((s) * 1e9)

/// CONVERT NANOSECONDS TO SECONDS
#define NS_TO_S(ns) ((ns) / 1e9)
/// CONVERT NANOSECONDS TO MILLISECONDS
#define NS_TO_MS(ns) ((ns) / 1e6)
/// CONVERT NANOSECONDS TO MICROSECONDS
#define NS_TO_US(ns) ((ns) / 1e3)

    /// STRUCTURE FOR STORING PGM IMAGE DATA
    typedef struct pgm_content_t
    {
        char pgmType[3];    /// PGM FORMAT TYPE (P2 OR P5)
        uint8_t *data;      /// ACTUAL IMAGE DATA
        uint32_t width;     /// IMAGE WIDTH
        uint32_t height;    /// IMAGE HEIGHT
        uint32_t max_value; /// MAXIMUM PIXEL VALUE
    } pgm_content_t;

    /// MAPS AN INTEGER VALUE FROM ONE RANGE TO ANOTHER
    /// @param input THE VALUE TO MAP
    /// @param input_start START OF INPUT RANGE
    /// @param input_end END OF INPUT RANGE
    /// @param output_start START OF OUTPUT RANGE
    /// @param output_end END OF OUTPUT RANGE
    /// @return THE MAPPED VALUE
    uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

    /// MAPS AN INTEGER VALUE FROM ONE RANGE TO ANOTHER WITH FLOATING-POINT PRECISION
    /// @param input THE VALUE TO MAP
    /// @param input_start START OF INPUT RANGE
    /// @param input_end END OF INPUT RANGE
    /// @param output_start START OF OUTPUT RANGE
    /// @param output_end END OF OUTPUT RANGE
    /// @return THE MAPPED VALUE WITH PRESERVED DECIMAL PRECISION
    uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

    /// GET CURRENT TIME IN MILLISECONDS
    /// @return TIMESTAMP IN MILLISECONDS
    uint64_t millis();

    /// GET CURRENT TIME IN MICROSECONDS
    /// @return TIMESTAMP IN MICROSECONDS
    uint64_t micros();

    /// GET CURRENT TIME IN NANOSECONDS
    /// @return TIMESTAMP IN NANOSECONDS
    uint64_t nanos();

    /// DUMPS THE CORTEX' CONTENT TO A FILE.
    /// THE FILE IS CREATED IF NOT ALREADY PRESENT, OVERWRITTEN OTHERWISE.
    /// @param cortex THE CORTEX TO BE WRITTEN TO FILE.
    /// @param file_name THE DESTINATION FILE TO WRITE THE CORTEX TO.
    void c2d_to_file(unk_cortex2d_t *cortex, char *file_name);

    /// READS THE CONTENT FROM A FILE AND INITIALIZES THE PROVIDED CORTEX ACCORDINGLY.
    /// @param cortex THE CORTEX TO INIT FROM FILE.
    /// @param file_name THE FILE TO READ THE CORTEX FROM.
    void c2d_from_file(unk_cortex2d_t *cortex, char *file_name);

    /// @brief STORES THE STRING REPRESENTATION OF THE GIVEN CORTEX TO THE PROVIDED STRING [TARGET].
    /// @param cortex THE CORTEX TO INSPECT.
    /// @param result THE STRING TO FILL WITH CORTEX DATA.
    void c2d_to_string(unk_cortex2d_t *cortex, char *result);

    /// @brief SETS TOUCH FOR EACH NEURON IN THE PROVIDED CORTEX BY READING IT FROM A PGM MAP FILE.
    /// @param cortex THE CORTEX TO APPLY CHANGES TO.
    /// @param map_file_name THE PATH TO THE PGM MAP FILE TO READ.
    void c2d_touch_from_map(unk_cortex2d_t *cortex, char *map_file_name);

    /// @brief SETS INHEXC RATIO FOR EACH NEURON IN THE PROVIDED CORTEX BY READING IT FROM A PGM MAP FILE.
    /// @param cortex THE CORTEX TO APPLY CHANGES TO.
    /// @param map_file_name THE PATH TO THE PGM MAP FILE TO READ.
    void c2d_inhexc_from_map(unk_cortex2d_t *cortex, char *map_file_name);

    /// @brief SETS FIRE THRESHOLD FOR EACH NEURON IN THE PROVIDED CORTEX BY READING IT FROM A PGM MAP FILE.
    /// @param cortex THE CORTEX TO APPLY CHANGES TO.
    /// @param map_file_name THE PATH TO THE PGM MAP FILE TO READ.
    void c2d_fthold_from_map(unk_cortex2d_t *cortex, char *map_file_name);

#ifdef __cplusplus
}
#endif
#endif
