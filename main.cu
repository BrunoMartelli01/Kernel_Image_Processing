#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#include <ctime>
#include <cuda_runtime.h>

using namespace std;
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






int main(int argc, char** argv) {

    const char* input_file = "input.jpg";
    const char* output_file = "output.jpg";
    int mask_width = 3;

    cout<<"Caricamento immagine: "<< input_file<<endl;


    int width, height, channels;
    unsigned char* img = stbi_load(input_file, &width, &height, &channels, 0);
    if (!img) {
        cout<< "Errore nel caricamento dell'immagine: "<<input_file<<endl;
        return 1;
    }

    int  img_size = (width) * (height) * (channels);
    float* blur_img = (float*)malloc(img_size * sizeof(float));

    for (size_t i = 0; i < img_size; i++) {
        blur_img[i] = static_cast<float>(img[i]);
    }
    stbi_image_free(img);


    cout<< "Immagine caricata: "<< width <<"x"<< height<<" con "<<  channels<<" canali"<<endl;

    int image_size = width * height * channels * sizeof(float);
    int mask_size = mask_width * mask_width * sizeof(float);

    float* sequential_output = (float*)malloc(image_size);
    float* mask = (float*)malloc(mask_size);



    mask[0] = 1.0f/9.0f; mask[1] = 1.0f/9.0f; mask[2] = 1.0f/9.0f;
    mask[3] = 1.0f/9.0f; mask[4] = 1.0f/9.0f; mask[5] = 1.0f/9.0f;
    mask[6] = 1.0f/9.0f; mask[7] = 1.0f/9.0f; mask[8] = 1.0f/9.0f;



    clock_t start = clock();
    sequential_convolution(blur_img, sequential_output, mask, width, height, channels, mask_width);
    clock_t end = clock();
    auto time = (end - start);
    cout<< "Tempo di esecuzione sequenziale: "<< time<<"ms"<<endl;


    unsigned char* out = (unsigned char*)malloc(width * height * channels);
    for (int i = 0; i < width * height * channels; i++) {
        out[i] = (unsigned char)(sequential_output[i]);
    }

    int success = stbi_write_jpg(output_file, width, height, channels, out, 100);
    if (!success) {
        cout<< "Errore nel salvataggio dell'immagine: "<< output_file<<endl;
    } else {
        cout<< "Immagine salvata con successo: "<< output_file<<endl;
    }



    free(blur_img);
    free(img);
    free(sequential_output);
    free(mask);



    return 0;
}