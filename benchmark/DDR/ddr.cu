#include <cstdlib>
#include <iostream>
#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA核函数：计算数组之和
__global__ void sumArrayKernel(const int* __restrict__ input, int* __restrict__ output, int n) {
    extern __shared__ int sharedData[];  // 共享内存，用于存储局部和

    // 每个线程的全局索引
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化共享内存
    sharedData[tid] = (globalIdx < n) ? input[globalIdx] : 0;
    __syncthreads();  // 确保所有线程都完成了初始化

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();  // 确保所有线程都完成了当前步的归约
    }

    // 将每个块的局部和写入输出数组
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int idX, idY;

constexpr int array_size = 1024;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " idX idY" << std::endl;
        return 1;
    }

    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    // 分配设备内存
    int *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(int) * array_size);  // 输入数组
    cudaMalloc(&d_output, sizeof(int) * ((array_size + 255) / 256));  // 每个块输出一个局部和

    // 接收数组数据（直接使用设备内存）
    receiveMessage(idX, idY, 0, 0, d_input, sizeof(int) * array_size);

    // 定义块大小和网格大小
    int blockSize = 256;
    int gridSize = (array_size + blockSize - 1) / blockSize;

    // 调用核函数
    sumArrayKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, array_size);

    // 在设备上计算最终总和
    int *d_finalSum;
    cudaMalloc(&d_finalSum, sizeof(int));  // 存储最终结果的设备内存
    sumArrayKernel<<<1, blockSize, blockSize * sizeof(int)>>>(d_output, d_finalSum, gridSize);

    // 发送最终结果（直接使用设备内存）
    sendMessage(0, 0, idX, idY, d_finalSum, sizeof(int));

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_finalSum);

    return 0;
}