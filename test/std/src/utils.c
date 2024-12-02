#include "utils.h"

/// SKIPS COMMENT LINES AND WHITESPACE IN PGM FILES
/// @param fp FILE POINTER TO THE PGM FILE
void ignoreComments(FILE *fp)
{
    int ch;
    char line[100];
    // IGNORE ANY BLANK LINES
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
    {
    }
    // RECURSIVELY IGNORE COMMENTS
    // IN A PGM IMAGE COMMENTED LINES START WITH '#'
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

/// READS A PGM IMAGE FILE INTO THE PROVIDED STRUCTURE
/// @param pgm POINTER TO PGM STRUCTURE TO STORE THE IMAGE
/// @param filename PATH TO THE PGM FILE
void pgm_read(pgm_content_t *pgm, const char *filename)
{
    // OPEN THE IMAGE FILE IN READ MODE
    FILE *pgmfile = fopen(filename, "r");
    // IF FILE DOES NOT EXIST, THEN RETURN
    if (pgmfile == NULL)
    {
        printf("FILE DOES NOT EXIST: %s\n", filename);
        return;
    }
    ignoreComments(pgmfile);
    // READ FILE TYPE
    fscanf(pgmfile, "%s", pgm->pgmType);
    ignoreComments(pgmfile);
    // READ IMAGE DIMENSIONS
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));
    ignoreComments(pgmfile);
    // READ MAXIMUM PIXEL VALUE
    fscanf(pgmfile, "%u", &(pgm->max_value));
    ignoreComments(pgmfile);
    // ALLOCATE MEMORY FOR IMAGE DATA
    pgm->data = (uint8_t *)malloc(pgm->width * pgm->height * sizeof(uint8_t));

    // READ IMAGE DATA BASED ON PGM TYPE
    if (!strcmp(pgm->pgmType, "P2"))
    {
        // READ ASCII FORMAT (P2)
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
        // READ BINARY FORMAT (P5)
        fread(pgm->data, sizeof(uint8_t), pgm->width * pgm->height, pgmfile);
    }
    else
    {
        printf("UNSUPPORTED PGM FILE TYPE!\n");
        exit(EXIT_FAILURE);
    }
    fclose(pgmfile);
}

/// MAPS AN INTEGER VALUE FROM ONE RANGE TO ANOTHER
/// @param input THE VALUE TO MAP
/// @param input_start START OF INPUT RANGE
/// @param input_end END OF INPUT RANGE
/// @param output_start START OF OUTPUT RANGE
/// @param output_end END OF OUTPUT RANGE
/// @return THE MAPPED VALUE
uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end)
{
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}

/// MAPS AN INTEGER VALUE FROM ONE RANGE TO ANOTHER WITH FLOATING-POINT PRECISION
/// @param input THE VALUE TO MAP
/// @param input_start START OF INPUT RANGE
/// @param input_end END OF INPUT RANGE
/// @param output_start START OF OUTPUT RANGE
/// @param output_end END OF OUTPUT RANGE
/// @return THE MAPPED VALUE WITH PRESERVED DECIMAL PRECISION
uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end)
{
    double slope = ((double)output_end - (double)output_start) / ((double)input_end - (double)input_start);
    return (double)output_start + slope * ((double)input - (double)input_start);
}

/// GET CURRENT TIME IN MILLISECONDS
/// @return TIMESTAMP IN MILLISECONDS
uint64_t millis()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ms = S_TO_MS((uint64_t)ts.tv_sec) + NS_TO_MS((uint64_t)ts.tv_nsec);
    return ms;
}

/// GET CURRENT TIME IN MICROSECONDS
/// @return TIMESTAMP IN MICROSECONDS
uint64_t micros()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t us = S_TO_US((uint64_t)ts.tv_sec) + NS_TO_US((uint64_t)ts.tv_nsec);
    return us;
}

/// GET CURRENT TIME IN NANOSECONDS
/// @return TIMESTAMP IN NANOSECONDS
uint64_t nanos()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ns = S_TO_NS((uint64_t)ts.tv_sec) + (uint64_t)ts.tv_nsec;
    return ns;
}

/// DUMPS THE CORTEX' CONTENT TO A FILE.
/// THE FILE IS CREATED IF NOT ALREADY PRESENT, OVERWRITTEN OTHERWISE.
/// @param cortex THE CORTEX TO BE WRITTEN TO FILE.
/// @param file_name THE DESTINATION FILE TO WRITE THE CORTEX TO.
void c2d_to_file(unk_cortex2d_t *cortex, char *file_name)
{
    // OPEN OUTPUT FILE FOR BINARY WRITING
    FILE *out_file = fopen(file_name, "wb");
    if (out_file == NULL)
    {
        printf("CANNOT CREATE FILE: %s\n", file_name);
        return;
    }
    // WRITE CORTEX METADATA
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
    // WRITE ALL NEURONS
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(unk_neuron_t), 1, out_file);
        }
    }
    fclose(out_file);
}

/// READS THE CONTENT FROM A FILE AND INITIALIZES THE PROVIDED CORTEX ACCORDINGLY.
/// @param cortex THE CORTEX TO INIT FROM FILE.
/// @param file_name THE FILE TO READ THE CORTEX FROM.
void c2d_from_file(unk_cortex2d_t *cortex, char *file_name)
{
    // OPENS THE FILE FOR READING
    FILE *in_file = fopen(file_name, "rb");
    // READ CORTEX METADATA
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
    // READ ALL NEURONS
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

/// @brief STORES THE STRING REPRESENTATION OF THE GIVEN CORTEX TO THE PROVIDED STRING [TARGET].
/// @param cortex THE CORTEX TO INSPECT.
/// @param result THE STRING TO FILL WITH CORTEX DATA.
void c2d_to_string(unk_cortex2d_t *cortex, char *result)
{
    snprintf(result,
             256,
             "cortex(\n\twidth:%d\n\theight:%d\n\tnh_radius:%d\n\tpulse_window:%d\n\tsample_window:%d\n)",
             cortex->width,
             cortex->height,
             cortex->nh_radius,
             cortex->pulse_window,
             cortex->sample_window);
}

/// @brief SETS TOUCH FOR EACH NEURON IN THE PROVIDED CORTEX BY READING IT FROM A PGM MAP FILE.
/// @param cortex THE CORTEX TO APPLY CHANGES TO.
/// @param map_file_name THE PATH TO THE PGM MAP FILE TO READ.
void c2d_touch_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;
    pgm_read(&pgm_content, map_file_name);
    // VERIFY DIMENSION MATCH
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
        printf("\nERROR: MAP DIMENSIONS DO NOT MATCH CORTEX DIMENSIONS\n");
        return;
    }
}

/// @brief SETS INHEXC RATIO FOR EACH NEURON IN THE PROVIDED CORTEX BY READING IT FROM A PGM MAP FILE.
/// @param cortex THE CORTEX TO APPLY CHANGES TO.
/// @param map_file_name THE PATH TO THE PGM MAP FILE TO READ.
void c2d_inhexc_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;
    // Read file.
    pgm_read(&pgm_content, map_file_name);
    // Make sure sizes are correct.
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
