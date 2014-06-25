#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <tmmintrin.h>

#include "box_blur.h"

static int image_width = 0;
static int image_height = 0;
static int stride = 0;

/* ffmpeg -i input.png -f rawvideo -pix_fmt rgb32 output.raw */
static uint8_t* read_raw( const char *name )
{
    FILE *file = fopen( name, "r" );
    uint8_t *raw = malloc( stride*(image_height+2)*4 );
    fseek( file, 0, SEEK_SET );
    for( int y = 0; y < image_height; y++ )
    {
        for ( int x = 0; x < image_width; x++ )
        {
            fread( &raw[((x+1)+(y+1)*stride)*4], sizeof(uint8_t), sizeof(uint8_t)*4, file );
        }
    }
    fclose( file );
    for( int y = 0; y < image_height; y++ )
    {
        memcpy( &raw[(y*stride)*4], &raw[(1+y*stride)*4], sizeof(uint8_t)*4 );
        memcpy( &raw[((image_width+1)+y*stride)*4], &raw[(image_width+y*stride)*4], sizeof(uint8_t)*4 );
    }
    memcpy( &raw[0], &raw[stride*4], sizeof(uint8_t)*4*stride );
    memcpy( &raw[((image_height+1)*stride)*4], &raw[((image_height)*stride)*4], sizeof(uint8_t)*4*stride );
    return raw;
}

/* ffmpeg -s 403x403 -f rawvideo -pix_fmt rgb32 -i hachidorii_blurred.raw hachidorii_blurred.png */
static void write_raw( const char *name, uint8_t *raw )
{
    FILE *file = fopen( name, "wb" );
    for( int y = 0; y < image_height; y++ )
    {
        fwrite( &raw[(1+(y+1)*stride)*4], 1, sizeof(uint8_t)*image_width*4, file );
    }
    fclose( file );
}

static void write_padded( const char *name, uint8_t *raw )
{
    FILE *file = fopen( name, "wb" );
    fwrite( raw, 1, stride*(image_height+2)*4, file );
    fclose( file );
}

static void grayscale( uint8_t *raw, uint8_t *grayed, int x, int y )
{
    uint8_t B = raw[(x + y*stride)*4];
    uint8_t G = raw[(x + y*stride)*4+1];
    uint8_t R = raw[(x + y*stride)*4+2];
    uint8_t average = ((9*B)+(92*G)+(27*R))/128;
    
    grayed[(x + y*stride)*4] = average;
    grayed[(x + y*stride)*4+1] = average;
    grayed[(x + y*stride)*4+2] = average;
    grayed[(x + y*stride)*4+3] = raw[(x + y*stride)*4+3];
}

static void grayscale_simd( uint8_t *raw, uint8_t *grayed, int x, int y )
{
    static const uint8_t weights[16] __attribute__((aligned(16))) = {9,92,27,0,9,92,27,0,9,92,27,0,9,92,27,0};

    static const uint8_t alpha[16] __attribute__((aligned(16))) = {0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF};

    __asm__(
        "movdqu     %2,     %%xmm0 \n" /* 16 bytes (16x8bits, 128-bit), 4 pixels */
        "movdqu     %3,     %%xmm1 \n" /* 16 bytes (16x8bits, 128-bit), 4 pixels */
        "pmaddubsw  %4,     %%xmm0 \n" /* weigh (16x8bits) into (8x16bits, 128-bit) 32 bytes */
        "pmaddubsw  %4,     %%xmm1 \n" /* weigh (16x8bits) into (8x16bits, 128-bit) 32 bytes */
        "phaddw     %%xmm1, %%xmm0 \n" /* add two (8x16bits) into 8 averages (8x16bits) */
        "psraw      $7,     %%xmm0 \n" /* shift all (8x16bits) right by 7 bits ie. /128 */
        "packuswb   %%xmm0, %%xmm0 \n" /* pack down into (8x8bits) by saturaion */
        "punpcklbw  %%xmm0, %%xmm0 \n" /* interleave xmm0 */
        "movdqa     %%xmm0, %%xmm1 \n" /* save result into xmm1 */
        "punpcklbw  %%xmm0, %%xmm0 \n" /* repeat left side into xmm0 */
        "punpckhbw  %%xmm1, %%xmm1 \n" /* repeat right side into xmm1 */
        "por        %5,     %%xmm0 \n" /* add alpha into every fourth byte into xmm0 */
        "por        %5,     %%xmm1 \n" /* add alpha into every foruth byte into xmm1 */
        "movdqu     %%xmm0, %0     \n" /* move xmm0 into grayed[(x + y*stride)*4] */
        "movdqu     %%xmm1, %1     \n" /* move xmm1 into grayed[(x + y*stride)*4+16] */
        :"=m"(grayed[(x + y*stride)*4]),
        "=m"(grayed[(x + y*stride)*4+16])
        :"m"(raw[(x + y*stride)*4]),
        "m"(raw[(x + y*stride)*4+16]),
        "m"(*weights),
        "m"(*alpha)
    );

    //printf("raw:    %d %d %d\n", raw[(x + y*stride)*4], raw[(x + y*stride)*4+1], raw[(x + y*stride)*4+2]);
    //printf("grayed: %d %d %d\n", grayed[(x + y*stride)*4], grayed[(x + y*stride)*4+1], grayed[(x + y*stride)*4+2]);
}

static uint8_t clamp( int n, uint8_t min, uint8_t max )
{
    if ( n < min )
    {
        return min;
    }
    if ( n > max )
    {
        return max;
    }
    return (uint8_t)n;
}

static void sharpen( uint8_t *raw, uint8_t *modified, int x, int y )
{
    for ( int z = 0; z < 3; z++ )
    {
        modified[(x + y*stride)*4+z]=
        clamp(((raw[(x + y*stride)*4+z]*5)
        -(raw[(x+1 + y*stride)*4+z]) /* E */
        -(raw[(x + (y+1)*stride)*4+z]) /* S */
        -(raw[(x-1 + y*stride)*4+z]) /* W */
        -(raw[(x + (y-1)*stride)*4+z])), 0, 255); /* N */
    }
    modified[(x + y*stride)*4+3] = 0xFF;
}

static void sharpen_simd( uint8_t *raw, uint8_t *modified, int x, int y )
{
    static const uint16_t mask[8] __attribute__((aligned(16))) = {5,5,5,1,5,5,5,1};
    static const uint8_t alpha[8] __attribute__((aligned(16))) = {0,0,0,0xFF,0,0,0,0xFF};

    /* Algorithm
     * 5(center)+-1(east)+-1(south)+-1(west)-+-1(north)
     * =5(center)-east-south-west-north
     * =5(center)-((east+south)+(west+north)) */
    __asm__(
        "pmovzxbw   %1,     %%xmm0 \n" /* Pack center pixels into 16-bit xmm0 */
        "pmovzxbw   %2,     %%xmm1 \n" /* Pack east pixels into xmm1 */
        "pmovzxbw   %3,     %%xmm2 \n" /* Pack south pixels into xmm2 */
        "pmovzxbw   %4,     %%xmm3 \n" /* Pack west pixels into xmm3 */
        "pmovzxbw   %5,     %%xmm4 \n" /* Pack north pixels into xmm4 */
        "pmullw     %6,     %%xmm0 \n" /* Multiply center pixels by 5 */
        "paddw      %%xmm2, %%xmm1 \n" /* East + South */
        "paddw      %%xmm4, %%xmm3 \n" /* West + North */
        "paddw      %%xmm3, %%xmm1 \n" /* (East + South) + (West + North) */
        "psubw      %%xmm1, %%xmm0 \n" /* Center*5 - ((East + South) + (West + North)) */
        "packuswb   %%xmm0, %%xmm0 \n" /* Pack sharpen back into 8-bit vector */
        "por        %7,     %%xmm0 \n" /* add alpha into every fourth byte into xmm2 */
        "movq       %%xmm0, %0     \n" /* Move average into modified */
        :"=m"(modified[(x + y*stride)*4])
        :"m"(raw[(x + y*stride)*4]),
        "m"(raw[(x+1 + y*stride)*4]), /* E */
        "m"(raw[(x + (y+1)*stride)*4]), /* S */
        "m"(raw[(x-1 + y*stride)*4]), /* W */
        "m"(raw[(x + (y-1)*stride)*4]), /* N */
        "m"(*mask),
        "m"(*alpha)
    );
}

static void average_neighbors( uint8_t *raw, uint8_t *blurred, int x, int y )
{
    for ( int z = 0; z < 4; z++ )
    {
        blurred[(x + y*stride)*4+z]=
        (raw[(x + y*stride)*4+z]
        +raw[(x+1 + y*stride)*4+z] /* E */
        +raw[(x+1 + (y+1)*stride)*4+z] /* SE */
        +raw[(x + (y+1)*stride)*4+z] /* S */
        +raw[(x-1 + (y+1)*stride)*4+z] /* SW */
        +raw[(x-1 + y*stride)*4+z] /* W */
        +raw[(x-1 + (y-1)*stride)*4+z] /* NW */
        +raw[(x + (y-1)*stride)*4+z] /* N */
        +raw[(x+1 + (y-1)*stride)*4+z])/9; /* NE */
    }   
}

static void average_neighbors_simd( uint8_t *raw, uint8_t *blurred, int x, int y )
{
    static const uint16_t divide[8] __attribute__((aligned(16))) = {7282,7282,7282,7282,7282,7282,7282,7282};

    __asm__(
        "pmovzxbw   %1,     %%xmm0 \n" /* Pack an 8-bit vector into a 16-bit vector */
        "pmovzxbw   %2,     %%xmm1 \n" /* This allows us to perform operations greater than 0xFF or 255 */
        "pmovzxbw   %3,     %%xmm2 \n"
        "pmovzxbw   %4,     %%xmm3 \n"
        "pmovzxbw   %5,     %%xmm4 \n"
        "pmovzxbw   %6,     %%xmm5 \n"
        "pmovzxbw   %7,     %%xmm6 \n"
        "pmovzxbw   %8,     %%xmm7 \n"
        "pmovzxbw   %9,     %%xmm8 \n"
        "movdqa     %10,    %%xmm9 \n" /* Move an aligned double quadword into xmm9 */
        "paddw      %%xmm0, %%xmm1 \n" /* Vertically add xmm0 and xmm1 */
        "paddw      %%xmm2, %%xmm3 \n" /* Here we add up all the values of 2 pixels for BGRA */
        "paddw      %%xmm4, %%xmm5 \n" /* Parallel add xmm0-8 so we don't have any dependencies */
        "paddw      %%xmm6, %%xmm7 \n"
        "paddw      %%xmm8, %%xmm1 \n"
        "paddw      %%xmm1, %%xmm3 \n"
        "paddw      %%xmm5, %%xmm7 \n"
        "paddw      %%xmm3, %%xmm7 \n" /* xmm7 now contains all the sums */
        "pmulhuw    %%xmm9, %%xmm7 \n" /* Divide by 9 by use a multiply and having it wrap around (faster!) */
        "packuswb   %%xmm7, %%xmm7 \n" /* Pack 16-bit vector into 8-bit vector */
        "movq       %%xmm7, %0     \n" /* Move average into blurred */
        :"=m"(blurred[(x + y*stride)*4])
        :"m"(raw[(x + y*stride)*4]),
        "m"(raw[(x+1 + y*stride)*4]), /* E */
        "m"(raw[(x+1 + (y+1)*stride)*4]), /* SE */
        "m"(raw[(x + (y+1)*stride)*4]), /* S */
        "m"(raw[(x-1 + (y+1)*stride)*4]), /* SW */
        "m"(raw[(x-1 + y*stride)*4]), /* W */
        "m"(raw[(x-1 + (y-1)*stride)*4]), /* NW */
        "m"(raw[(x + (y-1)*stride)*4]), /* N */
        "m"(raw[(x+1 + (y-1)*stride)*4]), /* NE */
        "m"(*divide)
    );
}

int main( int agrc, char** argv )
{
    uint8_t *raw;
    char *filename = argv[1];
    image_width = atoi( argv[2] );
    image_height = atoi( argv[3] );
    printf( "%s %d %d\n", filename, image_width, image_height );
    stride = image_width+2;
    raw = read_raw( filename );
    write_padded( "padded.raw", raw );
    uint8_t *blurred = malloc( stride*(image_height+2)*4 );

    clock_t start = clock();
    /* Box Blur applied 3 times is extremely similar to Gaussian blur */
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            average_neighbors( raw, blurred, x, y );
        }
    }
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            average_neighbors( blurred, blurred, x, y );
        }
    }
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            average_neighbors( blurred, blurred, x, y );
        }
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "Regular box blur took %f seconds\n", seconds );
    write_raw( "blurred.raw", blurred );

    start = clock();
    memset( blurred, 0, stride*(image_height+2)*4 );
    /* Box Blur applied 3 times is extremely similar to Gaussian blur */
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x+=2 )
        {
            average_neighbors_simd( raw, blurred, x, y );
        }
    }
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x+=2 )
        {
            average_neighbors_simd( blurred, blurred, x, y );
        }
    }
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x+=2 )
        {
            average_neighbors_simd( blurred, blurred, x, y );
        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "SIMD blur took %f seconds\n", seconds );
    write_raw( "blurred_simd.raw", blurred );

    start = clock();
    memset( blurred, 0, stride*(image_height+2)*4 );
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            grayscale( raw, blurred, x, y );
        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "grayscale took %f seconds\n", seconds );
    write_raw( "grayscale.raw", blurred );

    start = clock();
    memset( blurred, 0, stride*(image_height+2)*4 );
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x+=8 )
        {
            grayscale_simd( raw, blurred, x, y );
        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "grayscale SIMD took %f seconds\n", seconds );
    write_raw( "grayscale_simd.raw", blurred );

    start = clock();
    memset( blurred, 0, stride*(image_height+2)*4 );
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            sharpen( raw, blurred, x, y );
        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "sharpen took %f seconds\n", seconds );
    write_raw( "sharpen.raw", blurred );

    start = clock();
    memset( blurred, 0, stride*(image_height+2)*4 );
    for ( int y = 0; y < image_height+2; y++ )
    {
        for ( int x = 0; x < stride; x++ )
        {
            sharpen_simd( raw, blurred, x, y );
        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf( "sharpen SIMD took %f seconds\n", seconds );
    write_raw( "sharpen_simd.raw", blurred );

    free( raw );
    free( blurred );
    return 0;
}