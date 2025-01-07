#include <immintrin.h> // for AVX2 intrinsics
#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
/*
Reproduce DeepGEMM (W2A2)
Note that instead of reordering weights, I shifted the input,
so only one LUT is needed
*/

// LUT size for 2-bit quantization
constexpr int LUT_SIZE = 16;
constexpr int VECTOR_SIZE = 256 / 8; // 256-bit vector, 32 elements for 8-bit
constexpr size_t NUM_ELEMENTS = 32;

const int8_t predefined_weights[4] = {-1, 0, 1, 2};
const int8_t predefined_activations[4] = {0, 1, 2, 3};

void generateRandomData(uint8_t* activations, uint8_t* weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> activation_dist(0, 255); // random 4-activations indices
    std::uniform_int_distribution<int> weight_dist(0, 255);     // random 4-weights indices

    // weight table
    uint8_t values[256]; // 2-bit index fills in 8 bits

    // initialize with values 0 to 255
    std::iota(values, values + 256, 0);

    // fill in weight and activation arrays
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        activations[i] = values[activation_dist(gen)]; // generate 0~255 randomly
        weights[i] = values[weight_dist(gen)]; // generate 0~255 randomly
    }
}

// generate LUT for precomputed products
void generateLUT(int8_t lut[LUT_SIZE], const int8_t weights[4], const int8_t activations[4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            lut[i * 4 + j] = weights[i] * activations[j];
        }
    }
}

int32_t deepgemmAVX2(const uint8_t* activations, const uint8_t* weights, size_t length) {
    // lookup table for 2-bit quantization
    alignas(32) int8_t lut[LUT_SIZE] = {0};

    // generate LUT
    generateLUT(lut, predefined_weights, predefined_activations);

    // load LUT into vector registers
    __m256i lut_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(lut));

    // 將 vec 的前 128 bits 複製到後 128 bits，因為 _mm256_shuffle_epi8 只在 128-bit lanes 查找
    lut_vec = _mm256_permute2x128_si256(lut_vec, lut_vec, 0x00);
    
    // test
    alignas(32) int8_t lut_vec_array[VECTOR_SIZE];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lut_vec_array), lut_vec);

    std::cout << "LUT Table: ";
    for (int i = 0; i < VECTOR_SIZE / 2; ++i) { // the other half is the duplicated table
        std::cout << static_cast<int>(lut_vec_array[i]) << " ";
    }
    std::cout << std::endl;

    __m256i result = _mm256_setzero_si256(); // accumulator for results

    for (size_t i = 0; i < length; i += VECTOR_SIZE) {
        // load activations and weights
        __m256i act_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&activations[i]));
        __m256i wt_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&weights[i]));

        for (int shift = 0; shift < 8; shift += 2) {
            // extract 2-bit indices (_mm256_set1_epi8(0x03) is the mask to extract 32 2-bit numbers)
            __m256i act_index = _mm256_and_si256(_mm256_srli_epi16(act_vec, shift), _mm256_set1_epi8(0x03));
            __m256i wt_index = _mm256_and_si256(_mm256_srli_epi16(wt_vec, shift), _mm256_set1_epi8(0x03));

            // combine indices for LUT lookup
            __m256i combined_index = _mm256_or_si256(act_index, _mm256_slli_epi16(wt_index, 2));
            // __m256i combined_index = _mm256_or_si256(wt_index, _mm256_slli_epi16(act_index, 2));

            // LUT lookup
            __m256i lut_values = _mm256_shuffle_epi8(lut_vec, combined_index); // shuffle between 128-lane!!!

            /// test
            // print indices and lookup results
            alignas(32) int8_t act_index_array[VECTOR_SIZE];
            alignas(32) int8_t wt_index_array[VECTOR_SIZE];
            alignas(32) int8_t combined_index_array[VECTOR_SIZE];
            alignas(32) int8_t lut_values_array[VECTOR_SIZE];

            _mm256_store_si256(reinterpret_cast<__m256i*>(act_index_array), act_index);
            _mm256_store_si256(reinterpret_cast<__m256i*>(wt_index_array), wt_index);
            _mm256_store_si256(reinterpret_cast<__m256i*>(combined_index_array), combined_index);
            _mm256_store_si256(reinterpret_cast<__m256i*>(lut_values_array), lut_values);

        //     std::cout << "Shift: " << shift << std::endl;
        //     std::cout << "act_index: ";
        //     for (int j = 0; j < VECTOR_SIZE; ++j) {
        //         std::cout << static_cast<int>(act_index_array[j]) << " ";
        //     }
        //     std::cout << std::endl;

        //     std::cout << "wt_index: ";
        //     for (int j = 0; j < VECTOR_SIZE; ++j) {
        //         std::cout << static_cast<int>(wt_index_array[j]) << " ";
        //     }
        //     std::cout << std::endl;

        //     std::cout << "combined_index: ";
        //     for (int j = 0; j < VECTOR_SIZE; ++j) {
        //         std::cout << static_cast<int>(combined_index_array[j]) << " ";
        //     }
        //     std::cout << std::endl;

        //     std::cout << "lut_values: ";
        //     for (int j = 0; j < VECTOR_SIZE; ++j) {
        //         std::cout << static_cast<int>(lut_values_array[j]) << " ";
        //     }
        //     std::cout << std::endl;
        //     /// test end

            // accumulate results
            result = _mm256_add_epi8(result, lut_values);
        }
    }

    // horizontal sum of the result vector
    int32_t final_result = 0;
    alignas(32) int8_t result_array[VECTOR_SIZE];
    _mm256_store_si256(reinterpret_cast<__m256i*>(result_array), result);
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        final_result += result_array[i];
    }

    return final_result;
}

int main() {

    alignas(32) uint8_t activations[NUM_ELEMENTS];
    alignas(32) uint8_t weights[NUM_ELEMENTS];

    generateRandomData(activations, weights);

    std::cout << "Number of (activation, weight) to compute: " << NUM_ELEMENTS << std::endl;

    std::cout << "Activations (Each number represents 4 2-bit activations): \n";
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        std::cout << static_cast<int>(activations[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "Weights (Each number represents 4 2-bit weights): \n";
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        int8_t weight = weights[i] == 0? -1: weights[i] == 85? 0: 1;
        std::cout << static_cast<int>(weights[i]) << " ";
    }
    std::cout << std::endl;

    int32_t result = deepgemmAVX2(activations, weights, NUM_ELEMENTS);

    std::cout << "Dot product result: " << result << std::endl;

    return 0;
}
