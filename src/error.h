#ifndef __UNKNOWN_ERROR__
#define __UNKNOWN_ERROR__

typedef enum
{
    UNK_ERROR_NONE = 0, // No error occurred.
    UNK_ERROR_NH_RADIUS_TOO_BIG = 1, // The neighborhood radius is too big for the given cortex.
    UNK_ERROR_FILE_DOES_NOT_EXIST = 2, // The file does not exist.
    UNK_ERROR_FILE_SIZE_WRONG = 3, // The file size does not match the cortex size.
    UNK_ERROR_FAILED_ALLOC = 4, // Failed to allocate memory.
    UNK_ERROR_CORTEX_UNALLOC = 5, // The cortex is not allocated.
    UNK_ERROR_SIZE_WRONG = 6, // The cortex size is wrong.
    UNK_ERROR_INVALID_ARGUMENT = 7, // The argument is invalid.
    UNK_ERROR_EXTERNAL_CAUSES = 8 // The error is caused by external factors.
} unk_error_code_t;

#endif
