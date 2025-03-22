#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <string>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)


static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err)
              << " (" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}



void sequential_convolution(const float* input,float* output,const float* mask,int width,int height,int channels,int mask_width) {
    int mask_radius = mask_width / 2;
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0.0f;
                for (int i = 0; i < mask_width; i++) {
                    for (int j = 0; j < mask_width; j++) {
                        int img_y = y + i - mask_radius;
                        int img_x = x + j - mask_radius;
                        if (img_y >= 0 && img_y < height && img_x >= 0 && img_x < width) {
                            sum += input[(img_y * width + img_x) * channels + c] * mask[i * mask_width + j];
                        }
                    }
                }
                output[(y * width + x) * channels + c] = sum;
            }
        }
    }
}

__global__ void parallel_convolutionV1(const float* input,  float* output, const float* __restrict__ mask,  int width,int height, int channels,  int Mask_width, int TILE_WIDTH ) {

    extern __shared__ float N_ds[];

    int Mask_radius = Mask_width / 2;
    int w = TILE_WIDTH + Mask_width - 1;

    int k = blockIdx.z;

    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;

    int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
    int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
    int src = (srcY * width + srcX) * channels + k;


    int destIdx = destY * w + destX;

    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        N_ds[destIdx] = input[src];
    } else {
        N_ds[destIdx] = 0;
    }


    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    destIdx = destY * w + destX;
    srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
    srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
    src = (srcY * width + srcX) * channels + k;

    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destIdx] = input[src];
        } else {
            N_ds[destIdx] = 0;
        }
    }

    __syncthreads();


    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    if (row < height && col < width) {
        for (int i = 0; i < Mask_width; i++) {
            for (int j = 0; j < Mask_width; j++) {
                int y = threadIdx.y + i;
                int x = threadIdx.x + j;
                sum += N_ds[y * w + x] * mask[i * Mask_width + j];
            }
        }


        output[(row * width + col) * channels + k] = sum;
    }
}



__global__ void parallel_convolutionV2(const float* input, float* output, const float* __restrict__ mask, int width, int height, int channels, int Mask_width, int TILE_WIDTH) {

    extern __shared__ float N_ds[];

    int Mask_radius = Mask_width / 2;
    int w = TILE_WIDTH + Mask_width - 1;
    int k = blockIdx.z;
    int numThreadsNeeded = w * w;
    int threadsPerBlock = blockDim.x * blockDim.y;
    int iterations = (numThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < iterations; iter++) {
        int loadIndex = iter * threadsPerBlock + threadIdx.y * blockDim.x + threadIdx.x;

        if (loadIndex < numThreadsNeeded) {
            int destY = loadIndex / w;
            int destX = loadIndex % w;
            int destIdx = destY * w + destX;

            int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
            int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
            int src = (srcY * width + srcX) * channels + k;

            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                N_ds[destIdx] = input[src];
            } else {
                N_ds[destIdx] = 0;
            }
        }
    }

    __syncthreads();

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;


    if (row < height && col < width) {
        float sum = 0.0f;


        for (int i = 0; i < Mask_width; i++) {
            for (int j = 0; j < Mask_width; j++) {
                int y = threadIdx.y + i;
                int x = threadIdx.x + j;


                if (y < w && x < w) {
                    sum += N_ds[y * w + x] * mask[i * Mask_width + j];
                }
            }
        }

        output[(row * width + col) * channels + k] = sum;
    }
}

bool verify_results(const float* cpu_result, const float* gpu_result, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            cout << "Divergenza al pixel " << i << ": CPU = " << cpu_result[i] << ", GPU = " << gpu_result[i] << endl;
            return false;
        }
    }
    return true;
}
int main(int argc, char** argv) {

    const char* input_file = "input.jpg";
    const char* output_file = "output.jpg";
    int mask_width = 5; //3;
    int tile_list[] = {8,12,16,20,24,28,32};
    string result_file = "tile , sq, par, speedup\n";
    const int num_iterations = 5;


    int mask_size = mask_width * mask_width * sizeof(float);
    float *mask = (float*) malloc(mask_size);
    for (int i = 0; i < mask_width * mask_width; i++) {
        mask[i] = 0;
    }
    mask[mask_width * mask_width / 2] = 1.0f;


    cout << "Caricamento immagine: " << input_file << endl;
    int width, height, channels;
    unsigned char* img = stbi_load(input_file, &width, &height, &channels, 0);
    if (!img) {
        cout << "Errore nel caricamento dell'immagine: " << input_file << endl;
        return 1;
    }

    int img_size = (width) * (height) * (channels);
    float* blur_img = (float*)malloc(img_size * sizeof(float));

    for (size_t i = 0; i < img_size; i++) {
        blur_img[i] = static_cast<float>(img[i]);
    }
    stbi_image_free(img);

    cout << "Immagine caricata: " << width << "x" << height << " con " << channels << " canali" << endl;

    int image_size = width * height * channels * sizeof(float);
    float* sequential_output = (float*)malloc(image_size);


    cout << "Esecuzione calcolo sequenziale (" << num_iterations << " iterazioni)..." << endl;
    clock_t s_total = 0;

    for (int iter = 0; iter < num_iterations; iter++) {
        clock_t s = clock();
        sequential_convolution(blur_img, sequential_output, mask, width, height, channels, mask_width);
        clock_t end = clock();
        s_total += (end - s);
        cout << "  Iterazione " << (iter + 1) << ": " << (end - s) << "ms" << endl;
    }

    float sequential_time = static_cast<float>(s_total) / num_iterations;
    cout << "Tempo medio di esecuzione sequenziale: " << sequential_time << "ms" << endl;


    for (int tile_width : tile_list) {
        cout << "---------------------------------------------------" << endl;
        float* parallel_output = (float*)malloc(image_size);

        float *cuda_input, *cuda_output, *cuda_mask;
        CUDA_CHECK_RETURN(cudaMalloc(&cuda_input, image_size));
        CUDA_CHECK_RETURN(cudaMalloc(&cuda_output, image_size));
        CUDA_CHECK_RETURN(cudaMalloc(&cuda_mask, mask_size));

        CUDA_CHECK_RETURN(cudaMemcpy(cuda_input, blur_img, image_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(cuda_mask, mask, mask_size, cudaMemcpyHostToDevice));

        dim3 blockDim(tile_width, tile_width);
        dim3 gridDim(
            (width + tile_width - 1) / tile_width,
            (height + tile_width - 1) / tile_width,
            channels
        );
        int shared_mem_width = tile_width + mask_width;
        int shared_mem_size = shared_mem_width * shared_mem_width * sizeof(float);

        cout << "Dimensione tile: " << tile_width << "x" << tile_width << endl;
        cout << "Dimensione maschera: " << mask_width << "x" << mask_width << endl;
        cout << "Dimensione memoria condivisa: " << shared_mem_size << " byte" << endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        float parallel_time_total = 0;

        for (int iter = 0; iter < num_iterations; iter++) {
            cudaEventRecord(start);
            parallel_convolutionV2<<<gridDim, blockDim, shared_mem_size>>>(
                cuda_input, cuda_output, cuda_mask, width, height, channels, mask_width, tile_width
            );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float parallel_time_iter;
            cudaEventElapsedTime(&parallel_time_iter, start, stop);
            parallel_time_total += parallel_time_iter;

            cout << "  Iterazione GPU " << (iter + 1) << ": " << parallel_time_iter << "ms" << endl;
        }

        float parallel_time = parallel_time_total / num_iterations;
        CUDA_CHECK_RETURN(cudaMemcpy(parallel_output, cuda_output, image_size, cudaMemcpyDeviceToHost));

        cout << "Tempo medio di esecuzione GPU: " << parallel_time << "ms" << endl;
        cout << "Speedup: " << sequential_time/parallel_time << "x" << endl;

        float tolerance = 1e-4;
        bool results_match = verify_results(sequential_output, parallel_output, width * height * channels, tolerance);
        if (results_match) {
            cout << "SUCCESSO: I risultati della GPU corrispondono alla CPU entro la tolleranza " << tolerance << endl;
        } else {
            cout << "ERRORE: I risultati della GPU non corrispondono alla CPU" << endl;
        }


        if (tile_width == tile_list[sizeof(tile_list)/sizeof(tile_list[0]) - 1]) {
            unsigned char* out = (unsigned char*)malloc(width * height * channels);
            for (int i = 0; i < width * height * channels; i++) {
                out[i] = (unsigned char)(parallel_output[i]);
            }

            int success = stbi_write_jpg(output_file, width, height, channels, out, 100);
            if (!success) {
                cout << "Errore nel salvataggio dell'immagine: " << output_file << endl;
            } else {
                cout << "Immagine salvata con successo: " << output_file << endl;
            }
            free(out);
        }

        result_file += to_string(tile_width) + ", " + to_string(sequential_time) + ", " + to_string(parallel_time) + ", " + to_string(sequential_time/parallel_time) + "\n";

        free(parallel_output);
        cudaFree(cuda_input);
        cudaFree(cuda_output);
        cudaFree(cuda_mask);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    free(blur_img);
    free(sequential_output);
    free(mask);

    std::ofstream file("results.csv");
    if (file.is_open()) {
        file << result_file;
        file.close();
        cout << "CSV scritto correttamente!" << endl;
    }
    return 0;
}