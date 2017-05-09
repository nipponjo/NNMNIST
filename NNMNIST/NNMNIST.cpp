// NNMNIST.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <list>

using namespace std;

unsigned char **training_images;
unsigned char  *training_data;
unsigned char **validation_images;
unsigned char  *validation_data;
unsigned char **test_images;
unsigned char  *test_data;


int sizes[] = { 784, 30, 10 };

const int layers = 3;

double** w[layers - 1];
double* b[layers - 1];

const int n = layers - 1;

const int eta = 0;

int reverseInt(int);
void read_mnist();
double* add(double*, double*, int);
void backprop(unsigned char x[784], unsigned char y, double** nabla_w[], double* nabla_b[]);
void SGD(int epochs, int mini_batch_size, double eta);
double dot(double*, double*, int);
double* dot(double**, double*, int, int);
double* dot_T(double**, double*, int, int);
double sigmoid(double);
double* sigmoid(double*, int);
double sigmoid_prime(double);
double* sigmoid_prime(double*, int);
double* cost_derivative(double* output_activations, double* y, int n);
double* hamard(double *a, double *b, int n);
void zero_wb(double** nabla_w[], double* nabla_b[]);
void rnd_wb(double** nabla_w[], double* nabla_b[]);
double* feedforward(unsigned char x_in[784]);



double* allocate_mem(double*** arr, int n, int m, bool zero = false)
{
	*arr = (double**)malloc(n * sizeof(double*));
	double *arr_data;
	arr_data = zero ? (double*)calloc(n * m, sizeof(double)) 
		: (double*)malloc(n * m * sizeof(double));		

	for (int i = 0; i < n; i++)
		(*arr)[i] = arr_data + i * m;
	return arr_data;
}

double* allocate_mem(double** arr, int n, bool zero = false)
{
	*arr = zero ? (double*)calloc(n, sizeof(double)) : (double*)malloc(n * sizeof(double));	
	return *arr;
}

void shuffle_images(unsigned char** images, unsigned char* labels, int n) {
	int r = 0;
	unsigned char* tmp_img;
	unsigned char tmp_lbl;
	for (int i = n - 1; i >= 0; i--) {
		r = rand() % (i + 1);

		tmp_img = images[i];
		images[i] = images[r];
		images[r] = tmp_img;

		tmp_lbl = labels[i];
		labels[i] = labels[r];
		labels[r] = tmp_lbl;
	}
}

int main()
{
	
	

	//int test[] = { 0,1,2,3,4,5,6,7,8,9 };

	//shuffle_array((void**)&test, 10);

	//for (int z = 0; z < 10; z++) {
	//	cout << test[z] << " ";
	//}

	//for (int i = 9; i >= 0; i--) {
	//	int r = rand() % (i+1);

	//	int tmp = test[i];
	//	test[i] = test[r];
	//	test[r] = tmp;

	//	for (int z = 0; z < 10; z++) {
	//		cout << test[z] << " ";
	//	}
	//	cout << endl;
	//}


	for (int l = 0; l < layers - 1; l++)
	{
		allocate_mem(&w[l], sizes[l + 1], sizes[l], 1);
		allocate_mem(&b[l], sizes[l], 1);
	}

	rnd_wb(w, b);

	//double x[784] = { 0 };
	//
	//backprop(x, 0);

	read_mnist();

	//shuffle_images(validation_images, validation_data, 10000);

	SGD(10, 10, 3.0);

	for (int j = 0; j < 10; j++) {
		double* res = feedforward(test_images[j]);
		cout << (int)test_data[j] << ": [";
		for (int i = 0; i < 10; i++) {
			cout << res[i] << '\t';
		}
		cout << "]" << endl;
	}

	for (int i = 0; i < 10; i++) {
		cout << (int)test_data[i];
		for (int j = 0; j < 748; j++) {
			cout << ((test_images[i][j] > 10) ? "0" : " ");
			if (!(j % 28)) cout << endl;
		}
	}


	system("pause");

	return 0;


}

int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255; c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist()
{
	ifstream file("C:/Users/CJ/Desktop/MNIST/t10k-images.idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0; file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_images = 0; file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		int n_rows = 0; file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		int n_cols = 0; file.read((char*)&n_cols, sizeof(n_cols));			
		n_cols = reverseInt(n_cols);
		
		test_images = new unsigned char*[number_of_images];
		for (int i = 0; i<number_of_images; i++)
		{						
			test_images[i] = (unsigned char*)malloc(sizeof(char)*n_rows*n_cols);
			file.read((char*)test_images[i], n_rows*n_cols);
		}		
	}

	ifstream file2("C:/Users/CJ/Desktop/MNIST/t10k-labels.idx1-ubyte", ios::binary);
	if (file2.is_open())
	{
		int magic_number = 0; file2.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_items = 0; file2.read((char*)&number_of_items, sizeof(number_of_items));
		number_of_items = reverseInt(number_of_items);

		test_data = new unsigned char[number_of_items];
		for (int i = 0; i<number_of_items; i++)
		{			
			file2.read((char*)&test_data[i], 1);
			//cout << (int)test_data[i];
		}
	}

	ifstream file3("C:/Users/CJ/Desktop/MNIST/train-images.idx3-ubyte", ios::binary);
	if (file3.is_open())
	{
		int magic_number = 0; file3.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_images = 0; file3.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		int n_rows = 0; file3.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		int n_cols = 0; file3.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		training_images = new unsigned char*[50000];
		validation_images = new unsigned char*[10000];
		for (int i = 0; i<50000; i++)
		{
			training_images[i] = (unsigned char*)malloc(sizeof(char)*n_rows*n_cols);
			file3.read((char*)training_images[i], n_rows*n_cols);
		}
		for (int i = 0; i<10000; i++)
		{
			validation_images[i] = (unsigned char*)malloc(sizeof(char)*n_rows*n_cols);
			file3.read((char*)validation_images[i], n_rows*n_cols);
		}
	}

	ifstream file4("C:/Users/CJ/Desktop/MNIST/train-labels.idx1-ubyte", ios::binary);
	if (file4.is_open())
	{
		int magic_number = 0; file4.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		int number_of_items = 0; file4.read((char*)&number_of_items, sizeof(number_of_items));
		number_of_items = reverseInt(number_of_items);

		training_data = new unsigned char[50000];
		validation_data = new unsigned char[10000];
		for (int i = 0; i<50000; i++)
		{
			file4.read((char*)&training_data[i], 1);			
		}
		for (int i = 0; i<10000; i++)
		{
			file4.read((char*)&validation_data[i], 1);			
		}	
	}	

}

void SGD(int epochs, int mini_batch_size, double eta) {
	
	double** nabla_w_tmp[n];
	double* nabla_b_tmp[n];

	double** nabla_w[n];
	double* nabla_b[n];

	for (int l = 0; l < n; l++)
	{
		allocate_mem(&nabla_w[l], sizes[l + 1], sizes[l], 1);
		allocate_mem(&nabla_w_tmp[l], sizes[l + 1], sizes[l], 1);
		allocate_mem(&nabla_b[l], sizes[l], 1);	
		allocate_mem(&nabla_b_tmp[l], sizes[l], 1);
	}

	for (int epo = 0; epo < epochs; epo++) {

		shuffle_images(training_images, training_data, 50000);
		//Teilbarkeit 50000
		for (int mb = 0; mb < 50000; mb += mini_batch_size) {

			for (int mb_i = 0; mb_i < mini_batch_size; mb_i++) {

				backprop(training_images[mb+mb_i], training_data[mb+mb_i], nabla_w_tmp, nabla_b_tmp);
				
				for (int l = 0; l < n; l++) {
					for (int r = 0; r < sizes[l + 1]; r++) {
						for (int c = 0; c < sizes[l]; c++) {
							nabla_w[l][r][c] += nabla_w_tmp[l][r][c];
						}
						nabla_b[l][r] += nabla_b_tmp[l][r];
					}
				}

			}

			for (int l = 0; l < n; l++) {
				for (int r = 0; r < sizes[l + 1]; r++) {
					for (int c = 0; c < sizes[l]; c++) {
						w[l][r][c] -= (eta/mini_batch_size)*nabla_w_tmp[l][r][c];
					}
					b[l][r] -= (eta/mini_batch_size)*nabla_b_tmp[l][r];
				}
			}

			zero_wb(nabla_w_tmp, nabla_b_tmp);

			}
		cout << "Epoch: " << epo << endl;

		for (int j = 0; j < 10; j++) {
			double* res = feedforward(test_images[j]);
			cout << (int)test_data[j] << ": [";
			for (int i = 0; i < 10; i++) {
				cout << res[i] << '\t';
			}
			cout << "]" << endl;
		}

	}

}

void rnd_wb(double** nabla_w[], double* nabla_b[]) {
	for (int l = 0; l < n; l++) {
		for (int r = 0; r < sizes[l + 1]; r++) {
			for (int c = 0; c < sizes[l]; c++) {
				nabla_w[l][r][c] = -0.5 + (double)rand()/RAND_MAX;
			}
			nabla_b[l][r] = -0.5 + (double)rand() / RAND_MAX;
		}
	}
}

void zero_wb(double** nabla_w[], double* nabla_b[]) {
	for (int l = 0; l < n; l++) {
		for (int r = 0; r < sizes[l + 1]; r++) {
			for (int c = 0; c < sizes[l]; c++) {
				nabla_w[l][r][c] = 0.0;
			}
			nabla_b[l][r] = 0.0;
		}
	}
}

void update_mini_batch() {



}

void backprop(unsigned char x_in[784], unsigned char y_in, double** nabla_w[], double* nabla_b[]) {
	
	//double** nabla_w[n];
	//double* nabla_b[n];

	double* zs[n];
	
	//for (int l = 0; l < n; l++)
	//{
	//	allocate_mem(&nabla_w[l], sizes[l+1], sizes[l], 1);
	//	allocate_mem(&nabla_b[l], sizes[l], 1);
	//	//allocate_mem(&z[l], sizes[l + 1]);
	//}

	double x[784];
	double y[10] = { 0.0 };

	for (int i = 0; i < 784; i++) {
		x[i] = x_in[i]/255.0;
	}
	y[(int)y_in] = 1.0;
	
	double* activation = x;
	double* activations[layers];

	activations[0] = x;
	
	//feedforward
	for (int l = 0; l < n; l++)
	{
		//for (int j = 0; j < sizes[l]; j++) {
			double* z = add(dot(w[l], activation, sizes[l+1], sizes[l]), b[l], sizes[l+1]);
			zs[l] = z;
			activation = sigmoid(z, sizes[l+1]);
			activations[l + 1] = activation;
		//}		
	}

	double* delta = hamard(cost_derivative(activations[n], y, sizes[n]),
		sigmoid_prime(zs[n-1], sizes[n]), sizes[n]);

	nabla_b[n-1] = delta; //Speicherallokation...
	for (int i = 0; i < sizes[n]; i++)
		for (int j = 0; j < sizes[n - 1]; j++)
			nabla_w[n-1][i][j] = delta[i] * activations[n - 1][j];

	for (int l = n-1; l >= 1; l--) {
		double *sp = sigmoid_prime(zs[l-1], sizes[l]);
		delta = hamard(dot_T(w[l], delta, sizes[l+1], sizes[l]), sp, sizes[l]);
		nabla_b[l - 1] = delta;

		for (int i = 0; i < sizes[l]; i++)
			for (int j = 0; j < sizes[l - 1]; j++)
				nabla_w[l - 1][i][j] = delta[i] * activations[l - 1][j];

	}	

}

double* feedforward(unsigned char x_in[784]) {

	double x[784];

	for (int i = 0; i < 784; i++) {
		x[i] = x_in[i] / 255.0;
	}

	double* activation = x;	

	for (int l = 0; l < n; l++)
	{
		//for (int j = 0; j < sizes[l]; j++) {
		double* z = add(dot(w[l], activation, sizes[l + 1], sizes[l]), b[l], sizes[l + 1]);		
		activation = sigmoid(z, sizes[l + 1]);		
		//}		
	}

	return activation;

}

double* hamard(double *a, double *b, int n) {
	double *hamard = new double[n];
	for (int i = 0; i < n; i++) {
		hamard[i] = a[i]*b[i];
	}
	return hamard;
}

double dot(double* a, double* b, int n) {
	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

double* dot(double** W, double* x, int n, int m) {
	double* sum;
	allocate_mem(&sum, n);	
	for (int i = 0; i < n; i++) {
		sum[i] = dot(W[i], x, m);
	}
	return sum;
}

double* dot_T(double** W, double* x, int n, int m) {
	double* sum;
	allocate_mem(&sum, m, 1);	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			sum[j] += W[i][j]*x[i];
		}		
	}
	return sum;
}

double* add(double* a, double* b, int n) {
	double* sum;
	allocate_mem(&sum, n);
	for (int i = 0; i < n; i++) {
		sum[i] = a[i] + b[i];
	}
	return sum;
}

double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

double* sigmoid(double* z, int n) {
	double *zs = new double[n];
	for (int i = 0; i < n; i++) {
		zs[i] = 1.0 / (1.0 + exp(-z[i]));
	}	
	return zs;
}

double sigmoid_prime(double z) {
	return sigmoid(z)*(1 + sigmoid(z));
}

double* sigmoid_prime(double* z, int n) {
	double *zsp = new double[n];
	for (int i = 0; i < n; i++) {
		zsp[i] = sigmoid(z[i])*(1 + sigmoid(z[i]));
	}
	return zsp;
}

double* cost_derivative(double* output_activations, double* y, int n) {
	double *cd = new double[n];
	for (int i = 0; i < n; i++) {
		cd[i] = output_activations[i] - y[i];
	}
	return cd;
}


