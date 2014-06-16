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
static uint8_t* read_raw( const char *name, long *length )
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
static void write_raw( const char *name, uint8_t *raw, long *length )
{
	FILE *file = fopen( name, "wb" );
	for( int y = 0; y < image_height; y++ )
	{
		fwrite( &raw[(1+(y+1)*stride)*4], 1, sizeof(uint8_t)*image_width*4, file );
	}
	fclose( file );
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

int main( int agrc, char** argv )
{
	uint8_t *raw;
	long length;
	char *filename = argv[1];
	image_width = atoi( argv[2] );
	image_height = atoi( argv[3] );
	printf("%s %d %d\n", filename, image_width, image_height);
	stride = image_width+2;
	raw = read_raw( filename, &length );
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
	printf("Regular box blur took %f seconds\n", seconds);

	write_raw( "me_blurred.raw", blurred, &length );
	
	free( raw );
	free( blurred );
	return 0;
}