
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h> 
#include <stdio.h>
#include <cstdlib>
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif
#define THREADS_PER_BLOCK 1024

using namespace std;
using namespace cv;

void showImg(Mat img);
float* extractHOGfeatures(Mat img, const int size);
float* MatF2Float(Mat m);
void trainNN(float* HOGfeatures, const int people, const int images, const int featureSize, const float factor, const int count);
float* prepareTrainingFeatures(const int people, const int images, const int size, const int featureSize);
float predict(float* nn, float* HOGfeatures, int nnidx, int idx, const int featureSize);
void testNN(const int people, const int images, const int size, const int featureSize);

cudaError_t GPU(float* HOGfeatures, float* nn, const int people, const int images, const int featureSize, const float factor, const int count);


__global__ void reduction(float* A, float* output, const int N) {
    int A_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x;
    __shared__ float data[THREADS_PER_BLOCK];
    if (A_index < N) {
        int n;
        if (blockIdx.x == gridDim.x - 1) {
            n = int((N - blockIdx.x * blockDim.x) / 2);
        }
        else {
            n = int(THREADS_PER_BLOCK / 2);
        }  
        if (index < n) {
            data[index] = A[A_index] + A[A_index + n];
            __syncthreads();
            n >>= 1;

            while (n > 0) {
                if (index < n) {
                    data[index] += data[index + n];
                }
                __syncthreads();
                n >>= 1;
            }

            if (index == 0) {
                output[blockIdx.x] = data[0];
            }
        }
    }
}

__global__ void multiply(float* features, float* nn, const int featureSize, const int people, const int index, const int nnidx, float* prediction) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < featureSize + 1) {
        if (idx % (featureSize + 1) < featureSize) {
            prediction[idx] = features[index + idx % featureSize] * nn[nnidx + idx];
        }
        else {
            prediction[idx] = nn[nnidx + idx];
        }
    }
}

__global__ void trainNNgpu(float *nn, float* features, const int index, const int featureSize, const float factor, float* prediction, float goal, const int nnidx)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < featureSize + 1) {
        float pred = exp(prediction[0]) / (exp(prediction[0]) + 1);
        if (idx < featureSize) {
            nn[nnidx + idx] -= factor * (2 * (pred - goal) * pred * (1 - pred) * features[index + idx]);
        }
        else {
            nn[nnidx + idx] -= factor * (2 * (pred - goal) * pred * (1 - pred));
        }   
    }
}


int main()
{
    srand(time(NULL));
    const int size = 64;

    const int featureSize = (size / 8 - 1) * (size / 8 - 1) * 36;
    const int people = 5;
    const int images = 9;
    const int testImages = 3;
    const float factor = 0.1;
    const int count = 10000;

    float* HOGfeatures = prepareTrainingFeatures(people, images, size, featureSize);

    trainNN(HOGfeatures, people, images, featureSize, factor, count);
    testNN(people, testImages, size, featureSize);

    float* nn = new float[(featureSize + 1) * people];

    for (int i = 0; i < (featureSize + 1) * people; i++) {
        nn[i] = float(rand()) / float(RAND_MAX) / 10;
    }

    cudaError_t cudaStatus = GPU(HOGfeatures, nn, people, images, featureSize, factor, count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    testNN(people, testImages, size, featureSize);

    return 0;
}


cudaError_t GPU(float* HOGfeatures, float* nn, const int people, const int images, const int featureSize, const float factor, const int count)
{
    float* dev_features = 0;
    float* dev_nn = 0;
    float* dev_prediction1 = 0;
    float* dev_prediction2 = 0;
    cudaError_t cudaStatus;
    fstream fs;
    int blocks = (featureSize + 1) / THREADS_PER_BLOCK + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_features, people * images * featureSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nn, people * (featureSize + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_prediction1, (featureSize + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_prediction2, blocks * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_features, HOGfeatures, people * images * featureSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_nn, nn, people * (featureSize + 1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch kernels on the GPU 

    for (int i = 0; i < count; i++) {
        int person = rand() % people;
        int image = rand() % images;
        int index = person * images * featureSize + image * featureSize;
        for (int k = 0; k < people; k++) {
            int nnidx = k * (featureSize + 1);
            float goal;
            if (k == person) {
                goal = 1;
            }
            else {
                goal = 0;
            }
            multiply << <blocks, THREADS_PER_BLOCK >> > (dev_features, dev_nn, featureSize, people, index, nnidx, dev_prediction1);
            reduction << <blocks, THREADS_PER_BLOCK >> > (dev_prediction1, dev_prediction2, featureSize + 1);
            reduction << <1, THREADS_PER_BLOCK >> > (dev_prediction2, dev_prediction2, blocks);
            trainNNgpu << <blocks, THREADS_PER_BLOCK >> > (dev_nn, dev_features, index, featureSize, factor, dev_prediction2, goal, nnidx);
        }
    }

    // cudaDeviceSynchronize waits for the kernels to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(nn, dev_nn, people * (featureSize + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    
    fs.open("nn.txt", ios::out | ios::trunc);

    if (!fs) {
        cerr << "unable to open file" << endl;
    }
    else {
        for (int i = 0; i < (featureSize + 1) * people; i++) {
            fs << nn[i] << " ";
        }

    }

    fs.close();

    delete[] nn;
    nn = NULL;

Error:
    cudaFree(dev_features);
    cudaFree(dev_nn);
    
    return cudaStatus;
}

float* extractHOGfeatures(Mat img, const int size) {
    resize(img, img, Size(size, size));

    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, 1);

    float* mg = MatF2Float(mag);
    float* ang = MatF2Float(angle);

    for (int i = 0; i < size * size; i++) {
        if (ang[i] >= 180) {
            ang[i] -= 180;
        }
    }

    const int hogSize = (size / 8) * (size / 8) * 9;
    float* hog = new float[hogSize];

    for (int i = 0; i < hogSize; i++) {
        hog[i] = 0;
    }

    for (int i = 0; i < int(size / 8); i++) {
        for (int j = 0; j < int(size / 8); j++) {
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    int index = (i * 8 + k) * size + j * 8 + l;
                    int idx1 = int(ang[index] / 20);
                    int idx2 = int((idx1 + 1) % 9);
                    float scalar2 = ang[index] - idx1 * 20;
                    float scalar1 = (20 - scalar2) / 20;
                    scalar2 /= 20;
                    idx1 += int(i * (size / 8 * 9) + j * 9);
                    idx2 += int(i * (size / 8 * 9) + j * 9);
                    hog[idx1] += scalar1 * mg[index];
                    hog[idx2] += scalar2 * mg[index];
                }
            }
        }
    }

    const int featureSize = (size / 8 - 1) * (size / 8 - 1) * 36;
    float* feature = new float[featureSize];

    for (int i = 0; i < size / 8 - 1; i++) {
        for (int j = 0; j < size / 8 - 1; j++) {
            float norm = 0;
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    for (int m = 0; m < 9; m++) {
                        int index = (i + k) * size / 8 * 9 + (j + l) * 9 + m;
                        norm += hog[index] * hog[index];
                    }
                }
            }
            norm = sqrt(norm);
            int idx = i * 36 * (size / 8 - 1) + j * 36;
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    for (int m = 0; m < 9; m++) {
                        int index = (i + k) * size / 8 * 9 + (j + l) * 9 + m;
                        feature[idx] = hog[index] / norm;
                        idx++;
                    }
                }
            }
        }
    }
    delete[] hog;
    hog = NULL;

    return feature;
}

float* MatF2Float(Mat m) {
    const int size = m.rows * m.cols;
    float* d = new float[size];

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            int idx = i * m.cols + j;
            d[idx] = m.at<float>(i, j);
        }
    }

    return d;
}

void showImg(Mat img) {
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", img);
    waitKey(0);
}


float* prepareTrainingFeatures(const int people, const int images, const int size, const int featureSize) {
    const int cSize = 13;
    char name[cSize];
    strcpy_s(name, "1/000001.jpg");
    float* HOGfeatures = new float[featureSize * people * images];

    for (int i = 1; i <= people; i++) {
        for (int j = 1; j <= images; j++) {
            name[0] = i + '0';
            name[7] = j + '0';
            Mat imgM = imread(name, IMREAD_GRAYSCALE);

            if (!imgM.data) {
                cout << "Could not open or find the image" << std::endl;
            }
            else {
                float* feature = extractHOGfeatures(imgM, size);
                for (int k = 0; k < featureSize; k++) {
                    HOGfeatures[((i - 1) * images + j - 1) * featureSize + k] = feature[k];
                }
                delete[] feature;
                feature = NULL;
            }
        }
    }

    return HOGfeatures;
}

void trainNN(float* HOGfeatures, const int people, const int images, const int featureSize, const float factor, const int count) {
    float* nn = new float[(featureSize + 1) * people];

    for (int i = 0; i < (featureSize + 1) * people; i++) {
        nn[i] = float(rand()) / float(RAND_MAX) / 10;
    }

    for (int i = 0; i < count; i++) {
        int person = rand() % people;
        int image = rand() % images;
        int idx = person * images * featureSize + image * featureSize;
        for (int k = 0; k < people; k++) {
            float goal;
            if (k == person) {
                goal = 1;
            }
            else {
                goal = 0;
            }
            int nnidx = k * (featureSize + 1);
            float prediction = predict(nn, HOGfeatures, nnidx, idx, featureSize);
            for (int j = 0; j < featureSize; j++) {
                nn[nnidx + j] -= factor * (2 * (prediction - goal) * prediction * (1 - prediction) * HOGfeatures[idx + j]);
            }
            nn[nnidx + featureSize] -= factor * (2 * (prediction - goal) * prediction * (1 - prediction));
        }
    }

    fstream fs;
    fs.open("nn.txt", ios::out | ios::trunc);

    if (!fs) {
        cerr << "unable to open file" << endl;
    }
    else {
        for (int i = 0; i < (featureSize + 1) * people; i++) {
            fs << nn[i] << " ";
        }

    }

    fs.close();

    delete[] nn;
    nn = NULL;

}

float predict(float* nn, float* HOGfeatures, int nnidx, int idx, const int featureSize) {
    float prediction = 0;
    for (int i = 0; i < featureSize; i++) {
        prediction += nn[nnidx + i] * HOGfeatures[idx + i];
    }
    prediction += nn[nnidx + featureSize];
    prediction = exp(prediction) / (exp(prediction) + 1);

    return prediction;
}

void testNN(const int people, const int images, const int size, const int featureSize) {
    const int cSize = 13;
    char name[cSize];
    strcpy_s(name, "1/000011.jpg");

    fstream fs;
    fs.open("nn.txt", ios::in);
    float* nn = new float[people * (featureSize + 1)];
    for (int i = 0; i < people * (featureSize + 1); i++) {
        fs >> nn[i];
    }

    for (int i = 1; i <= people; i++) {
        for (int j = 1; j <= images; j++) {
            name[0] = i + '0';
            name[7] = j + '0';
            Mat imgM = imread(name, IMREAD_GRAYSCALE);

            if (!imgM.data) {
                cout << "Could not open or find the image" << std::endl;
            }
            else {
                float* feature = extractHOGfeatures(imgM, size);
                cout << "person " << i << ":\t";
                float max = 0;
                int max_person = 0;
                for (int k = 0; k < people; k++) {
                    int nnidx = k * (featureSize + 1);
                    float prediction = predict(nn, feature, nnidx, 0, featureSize);
                    if (prediction > max) {
                        max = prediction;
                        max_person = k;
                    }
                    cout << k + 1 << " - " << prediction << "\t";
                }
                cout << "prediction - " << max_person + 1 << endl;

                delete[] feature;
                feature = NULL;
            }
        }
    }
    cout << endl;
}
