#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unknown/unknown.h>

int main(int argc, char **argv)
{
    unk_cortex_size_t cortex_width = 512;
    unk_cortex_size_t cortex_height = 256;
    unk_cortex_size_t input_width = 32;
    unk_cortex_size_t input_height = 1;
    uint32_t iterations_count = 1000;
    unk_nh_radius_t nh_radius = 2;
    unk_cortex2d_t *even_cortex;
    unk_cortex2d_t *odd_cortex;
    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    c2d_set_evol_step(even_cortex, 0x01U);
    c2d_set_pulse_mapping(even_cortex, UNK_PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(even_cortex, 24);
    char touchFileName[40];
    char inhexcFileName[40];
    snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    c2d_touch_from_map(even_cortex, touchFileName);
    c2d_inhexc_from_map(even_cortex, inhexcFileName);
    c2d_copy(odd_cortex, even_cortex);
    char cortex_string[100];
    c2d_to_string(even_cortex, cortex_string);
    printf("%s\n", cortex_string);
    unk_input2d_t *input;
    i2d_init(&input,
             (cortex_width / 2) - (input_width / 2),
             0,
             (cortex_width / 2) + (input_width / 2),
             input_height,
             UNK_DEFAULT_EXC_VALUE * 2,
             UNK_PULSE_MAPPING_FPROP);
    for (unk_cortex_size_t i = 0; i < input_width * input_height; i++)
    {
        input->values[i] = even_cortex->sample_window - 1;
    }
    uint64_t start_time = millis();
    for (uint32_t i = 0; i < iterations_count; i++)
    {
        unk_cortex2d_t *prev_cortex = i % 2 ? odd_cortex : even_cortex;
        unk_cortex2d_t *next_cortex = i % 2 ? even_cortex : odd_cortex;
        c2d_feed2d(prev_cortex, input);
        c2d_tick(prev_cortex, next_cortex);
        if ((i + 1) % 100 == 0)
        {
            uint64_t elapsed = millis() - start_time;
            double fps = i / (elapsed / 1000.0f);
            printf("Performed %d iterations in %lums; %.2f ticks per second\n", i + 1, elapsed, fps);
            c2d_to_file(even_cortex, (char *)"out/test.c2d");
        }
    }
    c2d_to_file(even_cortex, (char *)"out/test.c2d");
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);
    return 0;
}
