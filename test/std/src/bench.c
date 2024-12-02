#define _POSIX_C_SOURCE 199309L

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <unknown/unknown.h>

#define S_TO_MS(s) ((s)*1e3)
#define S_TO_US(s) ((s)*1e6)
#define S_TO_NS(s) ((s)*1e9)
#define NS_TO_S(ns) ((ns) / 1e9)
#define NS_TO_MS(ns) ((ns) / 1e6)
#define NS_TO_US(ns) ((ns) / 1e3)

typedef struct pgm_content_t
{
    char pgmType[3];
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t max_value;
} pgm_content_t;

void ignoreComments(FILE *fp)
{
    int ch;
    char line[100];
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
    {
    }
    if (ch == '#')
    {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    }
    else
    {
        fseek(fp, -1, SEEK_CUR);
    }
}

void pgm_read(pgm_content_t *pgm, const char *filename)
{
    FILE *pgmfile = fopen(filename, "r");
    if (pgmfile == NULL)
    {
        printf("File does not exist: %s\n", filename);
        return;
    }
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%s", pgm->pgmType);
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%u", &(pgm->max_value));
    ignoreComments(pgmfile);
    pgm->data = (uint8_t *)malloc(pgm->width * pgm->height * sizeof(uint8_t));
    if (!strcmp(pgm->pgmType, "P2"))
    {
        for (uint32_t y = 0; y < pgm->height; y++)
        {
            for (uint32_t x = 0; x < pgm->width; x++)
            {
                fscanf(pgmfile, "%hhu", &(pgm->data[IDX2D(x, y, pgm->width)]));
            }
        }
    }
    else if (!strcmp(pgm->pgmType, "P5"))
    {
        fread(pgm->data, sizeof(uint8_t), pgm->width * pgm->height, pgmfile);
    }
    else
    {
        printf("Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
    fclose(pgmfile);
}

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end)
{
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}

uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end)
{
    double slope = ((double)output_end - (double)output_start) / ((double)input_end - (double)input_start);
    return (double)output_start + slope * ((double)input - (double)input_start);
}

uint64_t millis()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ms = S_TO_MS((uint64_t)ts.tv_sec) + NS_TO_MS((uint64_t)ts.tv_nsec);
    return ms;
}

uint64_t micros()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t us = S_TO_US((uint64_t)ts.tv_sec) + NS_TO_US((uint64_t)ts.tv_nsec);
    return us;
}

uint64_t nanos()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ns = S_TO_NS((uint64_t)ts.tv_sec) + (uint64_t)ts.tv_nsec;
    return ns;
}

void c2d_to_file(unk_cortex2d_t *cortex, char *file_name)
{
    FILE *out_file = fopen(file_name, "wb");
    if (out_file == NULL)
    {
        printf("File does not exist: %s\n", file_name);
        return;
    }
    fwrite(&(cortex->width), sizeof(unk_cortex_size_t), 1, out_file);
    fwrite(&(cortex->height), sizeof(unk_cortex_size_t), 1, out_file);
    fwrite(&(cortex->ticks_count), sizeof(unk_ticks_count_t), 1, out_file);
    fwrite(&(cortex->evols_count), sizeof(unk_ticks_count_t), 1, out_file);
    fwrite(&(cortex->evol_step), sizeof(unk_ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_window), sizeof(unk_ticks_count_t), 1, out_file);
    fwrite(&(cortex->nh_radius), sizeof(unk_nh_radius_t), 1, out_file);
    fwrite(&(cortex->fire_threshold), sizeof(unk_neuron_value_t), 1, out_file);
    fwrite(&(cortex->recovery_value), sizeof(unk_neuron_value_t), 1, out_file);
    fwrite(&(cortex->exc_value), sizeof(unk_neuron_value_t), 1, out_file);
    fwrite(&(cortex->decay_value), sizeof(unk_neuron_value_t), 1, out_file);
    fwrite(&(cortex->syngen_chance), sizeof(unk_chance_t), 1, out_file);
    fwrite(&(cortex->synstr_chance), sizeof(unk_chance_t), 1, out_file);
    fwrite(&(cortex->max_tot_strength), sizeof(unk_syn_strength_t), 1, out_file);
    fwrite(&(cortex->max_syn_count), sizeof(unk_syn_count_t), 1, out_file);
    fwrite(&(cortex->inhexc_range), sizeof(unk_chance_t), 1, out_file);
    fwrite(&(cortex->sample_window), sizeof(unk_ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_mapping), sizeof(unk_pulse_mapping_t), 1, out_file);
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(unk_neuron_t), 1, out_file);
        }
    }
    fclose(out_file);
}

void c2d_from_file(unk_cortex2d_t *cortex, char *file_name)
{
    FILE *in_file = fopen(file_name, "rb");
    fread(&(cortex->width), sizeof(unk_cortex_size_t), 1, in_file);
    fread(&(cortex->height), sizeof(unk_cortex_size_t), 1, in_file);
    fread(&(cortex->ticks_count), sizeof(unk_ticks_count_t), 1, in_file);
    fread(&(cortex->evols_count), sizeof(unk_ticks_count_t), 1, in_file);
    fread(&(cortex->evol_step), sizeof(unk_ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_window), sizeof(unk_ticks_count_t), 1, in_file);
    fread(&(cortex->nh_radius), sizeof(unk_nh_radius_t), 1, in_file);
    fread(&(cortex->fire_threshold), sizeof(unk_neuron_value_t), 1, in_file);
    fread(&(cortex->recovery_value), sizeof(unk_neuron_value_t), 1, in_file);
    fread(&(cortex->exc_value), sizeof(unk_neuron_value_t), 1, in_file);
    fread(&(cortex->decay_value), sizeof(unk_neuron_value_t), 1, in_file);
    fread(&(cortex->syngen_chance), sizeof(unk_chance_t), 1, in_file);
    fread(&(cortex->synstr_chance), sizeof(unk_chance_t), 1, in_file);
    fread(&(cortex->max_tot_strength), sizeof(unk_syn_strength_t), 1, in_file);
    fread(&(cortex->max_syn_count), sizeof(unk_syn_count_t), 1, in_file);
    fread(&(cortex->inhexc_range), sizeof(unk_chance_t), 1, in_file);
    fread(&(cortex->sample_window), sizeof(unk_ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_mapping), sizeof(unk_pulse_mapping_t), 1, in_file);
    cortex->neurons = (unk_neuron_t *)malloc(cortex->width * cortex->height * sizeof(unk_neuron_t));
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            fread(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(unk_neuron_t), 1, in_file);
        }
    }
    fclose(in_file);
}

void c2d_touch_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;
    pgm_read(&pgm_content, map_file_name);
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height)
    {
        for (unk_cortex_size_t i = 0; i < cortex->width * cortex->height; i++)
        {
            cortex->neurons[i].max_syn_count =
                fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->max_syn_count);
        }
    }
    else
    {
        printf("\nc2d_touch_from_map file sizes do not match with cortex\n");
        return;
    }
}

void c2d_inhexc_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;
    pgm_read(&pgm_content, map_file_name);
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height)
    {
        for (unk_cortex_size_t i = 0; i < cortex->width * cortex->height; i++)
        {
            cortex->neurons[i].inhexc_ratio =
                fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->inhexc_range);
        }
    }
    else
    {
        printf("\nc2d_inhexc_from_map file sizes do not match with cortex\n");
        return;
    }
}

int main(int argc, char **argv)
{
    unk_cortex_size_t cortex_width = 512;
    unk_cortex_size_t cortex_height = 256;
    unk_cortex_size_t input_width = 32;
    unk_cortex_size_t input_height = 1;
    uint32_t iterations_count = 10000;
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
    printf("%s", cortex_string);
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
            printf("\nPerformed %d iterations in %llums; %.2f ticks per second", i + 1, elapsed, fps);
            c2d_to_file(even_cortex, (char *)"out/test.c2d");
        }
    }
    c2d_to_file(even_cortex, (char *)"out/test.c2d");
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);
    return 0;
}
