#include "utils.h"

void ignoreComments(FILE *fp)
{
    int ch;

    // Ignore any blank lines
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
    {
    }

    // Recursively ignore comments.
    // In a PGM image commented lines start with '#'.
    if (ch == '#')
    {
        char line[100];
        do {
            if (fgets(line, sizeof(line), fp) != NULL) {
                line[sizeof(line) - 1] = '\0'; // Ensure null-termination
            }
        } while (line[strlen(line) - 1] != '\n' && !feof(fp));
        ignoreComments(fp);
    }
    else
    {
        fseek(fp, -1, SEEK_CUR);
    }
}

unk_error_code_t pgm_read(pgm_content_t *pgm, const char *filename)
{
    // Open the image file in read mode.
    FILE *pgmfile = fopen(filename, "r");

    // If file does not exist, then return.
    if (pgmfile == NULL)
    {
        return UNK_ERROR_FILE_DOES_NOT_EXIST;
    }

    ignoreComments(pgmfile);

    // Read file type.
    if (fgets(pgm->pgmType, sizeof(pgm->pgmType), pgmfile) == NULL) {
        fclose(pgmfile);
        return UNK_ERROR_FILE_READ;
    }
    pgm->pgmType[strcspn(pgm->pgmType, "\n")] = '\0'; // Remove newline character if present

    ignoreComments(pgmfile);

    // Read data size.
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));

    ignoreComments(pgmfile);

    // Read maximum value.
    fscanf(pgmfile, "%u", &(pgm->max_value));

    ignoreComments(pgmfile);

    // Allocate memory to store data in the struct.
    pgm->data = (uint8_t *)malloc((unsigned long)pgm->width * pgm->height * sizeof(uint8_t));

    // Store data in the struct.
    if (!strcmp(pgm->pgmType, "P2"))
    {
        // Plain data.
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
        // Raw data.
        fread(pgm->data, sizeof(uint8_t), (size_t)pgm->width * pgm->height, pgmfile);
    }
    else
    {
        printf("WRONG_FILE_TYPE\n");
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(pgmfile);

    return UNK_ERROR_NONE;
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

unk_error_code_t c2d_to_file(unk_cortex2d_t *cortex, char *file_name)
{
    // Open output file if possible.
    int fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return UNK_ERROR_FILE_DOES_NOT_EXIST;
    }
    FILE *out_file = fdopen(fd, "wb");
    if (out_file == NULL) {
        close(fd);
        return UNK_ERROR_FILE_DOES_NOT_EXIST;
    }

    // Write cortex metadata to the output file.
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

    // Write all neurons.
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(unk_neuron_t), 1, out_file);
        }
    }

    fclose(out_file);
    return UNK_ERROR_NONE;
}

void c2d_from_file(unk_cortex2d_t *cortex, char *file_name)
{
    // Open output file if possible.
    FILE *in_file = fopen(file_name, "rb");

    // Read cortex metadata from the output file.
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

    // Read all neurons.
    cortex->neurons = (unk_neuron_t *)malloc((size_t)cortex->width * cortex->height * sizeof(unk_neuron_t));
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            fread(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(unk_neuron_t), 1, in_file);
        }
    }

    fclose(in_file);
}

unk_error_code_t c2d_touch_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;

    // Read file.
    unk_error_code_t error = pgm_read(&pgm_content, map_file_name);
    if (error)
    {
        return error;
    }

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height)
    {
        for (unk_cortex_size_t i = 0; i < cortex->width * cortex->height; i++)
        {
            cortex->neurons[i].max_syn_count = fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->max_syn_count);
        }
    }
    else
    {
        printf("\nFILE_SIZE_WRONG: C2D_TOUCH_FROM_MAP FILE SIZES DO NOT MATCH WITH CORTEX\n");
        return UNK_ERROR_FILE_SIZE_WRONG;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_inhexc_from_map(unk_cortex2d_t *cortex, char *map_file_name)
{
    pgm_content_t pgm_content;

    // Read file.
    unk_error_code_t error = pgm_read(&pgm_content, map_file_name);
    if (error)
    {
        return error;
    }

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height)
    {
        for (unk_cortex_size_t i = 0; i < cortex->width * cortex->height; i++)
        {
            cortex->neurons[i].inhexc_ratio = fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->inhexc_range);
        }
    }
    else
    {
        printf("\nFILE_SIZE_WRONG: C2D_TOUCH_FROM_MAP FILE SIZES DO NOT MATCH WITH CORTEX\n");
        return UNK_ERROR_FILE_SIZE_WRONG;
    }

    return UNK_ERROR_NONE;
}