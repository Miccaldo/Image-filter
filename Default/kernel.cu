
#include "cuda_runtime.h"
#include "CImg-2.5.0/CImg.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <time.h>
#include <windows.h> 
#include <stddef.h>

#define cimg_use_jpeg 1

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define WRITE_FLATTEN_MATRIX	false
#define GET_FLATTEN_MATRIX		true

#include "device_launch_parameters.h"
using namespace cimg_library;
using namespace std;


void filter(double *matrix, double *copy, double *mask, int width, int height, int channels, int maskLength, double normalize);
void writeOrGetMatrix(CImg<double> &image, double *matrix, bool type);
double getNormalize(double *mask, int size);
void blow(double *mask, int size);
void sharpen(double *mask);
void prominence(double *mask);
void gradient(double *mask);
int init();
int getMaskLength(int filterType);
void setFilter(double *mask, int filterType);
void laplace(double *mask);


__global__ void dev_filter(double *dev_matrix, double *dev_kopia, double *dev_mask, int width, int height, int channels, int maskLength, double *dev_sample, double normalize){
	int x = blockIdx.x*blockDim.x + threadIdx.x;		// przypisanie wspolprzednej do aktualnego watku
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y * gridDim.x;						// numer watka w 1D

	int sampleX = 0;
	int sampleY = 0;
	double result = 0;
	int divMaskLength = (maskLength / 2)-1;				// rozmiar maski liczony od srodka, czyli obrabianego piksela. Potrzeba do pêtli ktora mnozy maske oraz probke.

	if ((x != 0) && (y != 0) && (x != width - 1) && (y != height - 1)) {	// Petla glowna, nie brane sa pod uwage piksele po zewnetrznej stronie obrazka 

		for (int xx = -divMaskLength; xx < (maskLength - divMaskLength); xx++) {		// tworzona jest probka o wielkosci rownej masce, nastepnie kazdy element jest mnozony 
			for (int yy = -divMaskLength; yy < (maskLength - divMaskLength); yy++) {
				dev_sample[sampleY + sampleX * maskLength] = dev_kopia[offset + yy + xx * width];
				result += dev_sample[sampleY + sampleX * maskLength] * dev_mask[sampleY + sampleX * maskLength];
				sampleY++;
			}
			sampleX++;
			sampleY = 0;
		}
		if (normalize != 0) result /= normalize;		// normalizacja

		if (result >= 1) result = 1;
		if (result <= 0) result = 0;
		dev_matrix[offset] = result;
		__syncthreads();
	}
}


int main(void) {

	int filterType = init();
	int maskLength = getMaskLength(filterType);

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	int originalColor = 7;
	int CPUcolor = 9;
	int GPUcolor = 2;

	time_t start;
	time_t stop;
	double timeResult = 0;

	CImg<double> image("biedronka.jpg");
	CImgDisplay display(image, "Oryginal");

	int width = image.width();
	int height = image.height();
	int channels = image.spectrum();

	CImg<double> output(width, height, 1, channels);
	CImg<double> outputCPU(width, height, 1, channels);

	double *matrix = new double[width * height * channels];
	double *dev_matrix;
	double *kopia = new double[width * height * channels];
	double *dev_kopia;		

	double *mask = new double[maskLength * maskLength];
	double *dev_mask;
	double *sample = new double[maskLength * maskLength];
	double *dev_sample;


	writeOrGetMatrix(image, matrix, WRITE_FLATTEN_MATRIX);
	writeOrGetMatrix(image, kopia, WRITE_FLATTEN_MATRIX);

	setFilter(mask, filterType);

	double normalize = getNormalize(mask, maskLength);


	// ********************* CPU *************************
	// ===================================================

	start = clock();

	filter(matrix, kopia, mask, width, height, channels, maskLength, normalize);

	stop = clock();
	timeResult = (double)(stop - start) / CLOCKS_PER_SEC;

	SetConsoleTextAttribute(hConsole, CPUcolor);
	cout << " CZAS CPU: ";
	SetConsoleTextAttribute(hConsole, originalColor);
	cout << timeResult << "s";
	cout << endl << endl;

	writeOrGetMatrix(outputCPU, matrix, GET_FLATTEN_MATRIX);

	CImgDisplay display2(outputCPU, "CPU");

	writeOrGetMatrix(image, matrix, WRITE_FLATTEN_MATRIX);
	writeOrGetMatrix(image, kopia, WRITE_FLATTEN_MATRIX);

	// ********************* GPU *************************
	// ===================================================

	int sizeX = width;
	int sizeY = height * channels;


	int TILE = 1;

	dim3 block(TILE, TILE);

	int grid_x = sizeX;
	int grid_y = sizeY;

	dim3 grid(grid_x, grid_y);

	// ************* ALOKACJA PAMIECI NA GPU *************
	// ===================================================

	HANDLE_ERROR(cudaMalloc((void**)&dev_matrix, width * height * channels * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_kopia, width * height * channels * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_mask, maskLength * maskLength * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_sample, maskLength * maskLength * sizeof(double)));

	// ************* KOPIOWANIE PAMIECI DO GPU *************
	// =====================================================

	HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix, width * height * channels * sizeof(double), cudaMemcpyHostToDevice));	// kopiowanie do GPU
	HANDLE_ERROR(cudaMemcpy(dev_kopia, matrix, width * height * channels * sizeof(double), cudaMemcpyHostToDevice));	// kopiowanie do GPU
	HANDLE_ERROR(cudaMemcpy(dev_mask, mask, maskLength * maskLength * sizeof(double), cudaMemcpyHostToDevice));			// kopiowanie do GPU
	HANDLE_ERROR(cudaMemcpy(dev_sample, sample, maskLength * maskLength * sizeof(double), cudaMemcpyHostToDevice));			// kopiowanie do GPU

	// **************** WYWOLANIE KERNELA ******************
	// =====================================================

	start = clock();

	dev_filter <<<grid, block >>> (dev_matrix, dev_kopia, dev_mask, width, height, channels, maskLength, dev_sample, normalize);

	stop = clock();
	timeResult = (double)(stop - start) / CLOCKS_PER_SEC;

	SetConsoleTextAttribute(hConsole, GPUcolor);
	cout << " CZAS GPU: ";
	SetConsoleTextAttribute(hConsole, originalColor);
	cout << timeResult << "s";
	cout << endl << endl;

	// ************* KOPIOWANIE PAMIECI DO CPU *************
	// =====================================================

	HANDLE_ERROR(cudaMemcpy(matrix, dev_matrix, width * height * channels * sizeof(double), cudaMemcpyDeviceToHost));	// kopiowanie z GPU do CPU


	writeOrGetMatrix(output, matrix, GET_FLATTEN_MATRIX);

	CImgDisplay display3(output, "GPU");

	while (!(display.is_closed() && display2.is_closed() && display3.is_closed())){
		display.wait();
		display2.wait();
		display3.wait();
	}

	// **************** ZWOLNIENIE PAMIECI *****************
	// =====================================================

	delete[] mask;
	delete[] matrix;
	delete[] kopia;
	delete[] sample;

	cudaFree(dev_matrix);
	cudaFree(dev_kopia);
	cudaFree(dev_mask);

	return 0;
}


void writeOrGetMatrix(CImg<double> &image, double *matrix, bool type) {

	int width = image.width();
	int height = image.height();
	int channels = image.spectrum();

	for (int c = 0; c < channels; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if(!type) matrix[x + y * width + c * width * height] = image(x, y, c)/255;
				else image(x, y, c) = matrix[x + y * width + c * width * height];
			}
		}
	}
}


double getNormalize(double *mask, int size) {

	double result = 0;

	for (int i = 0; i < size * size; i++) {
		result += mask[i];
	}
	return result;
}

void blow(double *mask, int size) {
	for (int i = 0; i < size * size; i++) {
		mask[i] = 1;
	}
}

void sharpen(double *mask) {

	mask[0] = 0;
	mask[1] = -1;
	mask[2] = 0;
	mask[3] = -1;
	mask[4] = 5;
	mask[5] = -1;
	mask[6] = 0;
	mask[7] = -1;
	mask[8] = 0;

}

void prominence(double *mask) {

	mask[0] = -1;
	mask[1] = 0;
	mask[2] = 1;
	mask[3] = -1;
	mask[4] = 1;
	mask[5] = 1;
	mask[6] = -1;
	mask[7] = 0;
	mask[8] = 1;
}
void gradient(double *mask) {

	mask[0] = 0;
	mask[1] = 0;
	mask[2] = 0;
	mask[3] = -1;
	mask[4] = 1;
	mask[5] = 0;
	mask[6] = 0;
	mask[7] = 0;
	mask[8] = 0;

}

void laplace(double *mask) {

	mask[0] = -1;
	mask[1] = -1;
	mask[2] = -1;
	mask[3] = -1;
	mask[4] = 8;
	mask[5] = -1;
	mask[6] = -1;
	mask[7] = -1;
	mask[8] = -1;

}

void filter(double *matrix, double *copy, double *mask, int width, int height, int channels, int maskLength, double normalize) {

	double result = 0;
	int sampleX = 0;
	int sampleY = 0;
	double divMaskLength = floor(maskLength / 2);
	int index = 0;
	double *sample = new double[maskLength * maskLength];
	int cnt = 0;
	double mnozenie = 0;

	for (int k = 0; k < 3; k++) {
		for (int i = divMaskLength; i < (height - divMaskLength); i++) {
			for (int j = divMaskLength; j < (width - divMaskLength); j++) {
				index = j + (i * width) + (k * width * height);
				for (int x = -divMaskLength; x < (maskLength - divMaskLength); x++) {
					for (int y = -divMaskLength; y < (maskLength - divMaskLength); y++) {
						sample[sampleY + sampleX * maskLength] = copy[index + y + x * width];
						mnozenie = (sample[sampleY + sampleX * maskLength] * mask[sampleY + sampleX * maskLength]);
						result = result + mnozenie;
						sampleY++;

					}
					sampleX++;
					sampleY = 0;
				}

				if(normalize != 0) result = result / normalize;
				if (result >= 1) result = 1;
				if (result <= 0) result = 0;
				matrix[index] = result;

				result = 0;
				sampleX = 0;
				sampleY = 0;
			}
		}
	}
	delete[] sample;
}


int init() {

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	int availableColor = 16;
	int originalColor = 7;
	int setColor = 14;
	int errorColor = 12;

	int maskLength;
	string text;
	int number;
	int filtersCount = 5;

	string filtres[5] = {
		" * (1) ROZMYCIE",
		" * (2) WYOSTRZENIE",
		" * (3) UWYDATNIENIE",
		" * (4) GRADIENT",
		" * (5) LAPLACE"
	};

	do {
		system("cls");

		SetConsoleTextAttribute(hConsole, availableColor);
		cout << " Dostepne filtry: " << endl;
		SetConsoleTextAttribute(hConsole, originalColor);
		cout << "---------------------------" << endl;

		cout << filtres[0] << endl;
		cout << filtres[1] << endl;
		cout << filtres[2] << endl;
		cout << filtres[3] << endl;
		cout << filtres[4] << endl;

		cout << "---------------------------";
		cout << endl << endl;

		cout << " Wprowadz numer: ";
		cin >> text;
		number = atoi(text.c_str());
		number -= 1;
		if (!(number > -1 && number < filtersCount)) {
			SetConsoleTextAttribute(hConsole, errorColor);
			cout << " error!";
			SetConsoleTextAttribute(hConsole, originalColor);
			Sleep(1000);
		}
		else {
			Sleep(250);
		}
	} while (!(number > -1 && number < filtersCount));

	cout << "---------------------------" << endl;

	cout << endl << endl;

	system("cls");
	SetConsoleTextAttribute(hConsole, availableColor);
	cout << " Dostepne filtry: " << endl;
	SetConsoleTextAttribute(hConsole, originalColor);
	cout << "---------------------------" << endl;

	for (int i = 0; i < filtersCount; i++) {
		if (i == number) {
			SetConsoleTextAttribute(hConsole, setColor);
			cout << filtres[i] << endl;
		}
		else {
			SetConsoleTextAttribute(hConsole, originalColor);
			cout << filtres[i] << endl;
		}
	}
	SetConsoleTextAttribute(hConsole, originalColor);

	cout << "---------------------------";
	cout << endl << endl;

	return number + 1;
}

int getMaskLength(int filterType) {
	switch (filterType) {
	case 1:
		return 5;
		break;
	default:
		return 3;
		break;
	}
}

void setFilter(double *mask, int filterType) {

	switch (filterType) {
	case 1:
		blow(mask, 5);
		break;
	case 2:
		sharpen(mask);
		break;
	case 3:
		prominence(mask);
		break;
	case 4:
		gradient(mask);
		break;
	case 5:
		laplace(mask);
		break;
	}
}


