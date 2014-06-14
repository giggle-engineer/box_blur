#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <tmmintrin.h>

#include "box_blur.h"

#define IMAGE_WIDTH 403
#define IMAGE_HEIGHT 403

/* ffmpeg -i input.png -f rawvideo -pix_fmt rgb32 output.raw */
static uint8_t* read_raw( const char *name, long *length )
{
	FILE *file = fopen( name, "r" );
	fseek( file, 0, SEEK_END );
	*length = ftell( file );
	uint8_t *raw = malloc( *length );
	fseek( file, 0, SEEK_SET );
	fread( raw, 1, *length, file );
	fclose( file );
	return raw;
}

/* ffmpeg -s 403x403 -f rawvideo -pix_fmt rgb32 -i hachidorii_blurred.raw hachidorii_blurred.png */
static void write_raw( const char *name, uint8_t *raw, long *length )
{
	FILE *file = fopen( name, "wb" );
	fwrite( raw, 1, *length, file );
	fclose( file );
}

static void average_neighbors( uint8_t *raw, uint8_t *blurred, int location )
{
	blurred[location]=
	(raw[location+4] /* E */
	+raw[location+IMAGE_WIDTH*4+4] /* SE */
	+raw[location+IMAGE_WIDTH*4] /* S */
	+raw[location+IMAGE_WIDTH*4-4] /* SW */
	+raw[location-4] /* W */
	+raw[location-IMAGE_WIDTH*4-4] /* NW */
	+raw[location-IMAGE_WIDTH*4] /* N */
	+raw[location-IMAGE_WIDTH*4+4])/8; /* NE */
}

int main( int agrc, char** argv )
{
	uint8_t *raw;
	long length;
	const char filename[] = "hachidorii.raw";
	raw = read_raw( filename, &length );
	uint8_t *blurred = malloc(length);

	for ( int i; i < length; i++ )
	{
		average_neighbors( raw, blurred, i );
	}

	write_raw( "hachidorii_blurred.raw", blurred, &length );
	free( raw );
	free( blurred );
	return 0;
}