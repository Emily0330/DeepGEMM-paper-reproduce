# DeepGEMM-paper-reproduce
Reproduce [this paper](https://arxiv.org/pdf/2304.09049) (DeepGEMM: Accelerated Ultra Low-Precision Inference on CPU Architectures using Lookup Tables).  
The paper's goal is to accelerate the low-bit inference on x86 CPU with the support of AVX2 ISA.
## Table of Contents
* [Implementation Details](#implementation-details)
* [Notes on reading the paper (論文導讀)](#notes-on-reading-the-paper-論文導讀)
    * [Motivation](#motivation)
    * [Contribution](#contribution)
    * [Constructing LUT](#constructing-lut)
    * [Packing & Unpacking](#packing--unpacking)
    * [LUT16 Example](#lut16-example)
    * [Experiments](#experiments)
    * [Conclusion](#conclusion)
## Implementation Details
* Implementation: 
    1. Pre-build the lookup table of the multiplication results of all combinations of (weight, activation).
    2. Use **SIMD intrinsics** to do 32 multiplications at the same time using 256-bit vector registers.
    3. Accumulate the results. (Because we're computing dot products)
* Bit-width
    * Activation: 2 bits
    * Weight: 2 bits
* Numerical range
    * Activation: {0,1,2,3}
    * Weight: {-1,0,1,2}
    * These values are pre-defined, they can be any values (but at most 4 different values). You can change it at the front of the `deepgemm.cpp`
    
        ```c++
        const int8_t predefined_weights[4] = {-1, 0, 1, 2};
        const int8_t predefined_activations[4] = {0, 1, 2, 3};
        ```

## Notes on reading the paper (論文導讀)

### Motivation
* Ultra low bit quantization presents an attractive option for reducing neural network inference costs.
*  However, ultra low-bit deep learning operators can not be efficiently executed on these devices because sub-8-bit instructions are not generally supported in the instruction sets of mainstream CPU architectures.  

### Contribution
* Propose a novel approach, **DeepGEMM**, for extremely low-bit computations on CPUs with SIMD support that **utilizes lookup tables to replace MAC operations**.

### Constructing LUT
* **Activations and weights are both quantized to 2 bits**. (only 4 kinds of activations/weights)
* Pre-compute the dot products of weight and activation and store them into LUT.
* $0010$ is the index to access the precomputed value of $\alpha_{00}\cdot\omega_{10}$
![image](https://hackmd.io/_uploads/HyZ8NhSS1e.png)

* DeepGEMM: packing => unpacking => lookup
### Packing & Unpacking
#### Packing
1. Original weights (FP) are quantized to ultra low-bit and ++cast to integer (??)++
2. Vectorize (ex. 2 or 4 bits => ? bits) (to be packed into higher precision data types)
    * 這張圖看起來左圖共 pack 4 個 2-bit weights ($\omega_{10},\omega_{01},\omega_{11},\omega_{01}$)
3. shift left (讓下一個 2-bit weights 可以從 lowest 2 bits 進來)
4. 跟下一個 vectorized weight 做 OR (把下一個 weight 塞進 weight vector)
5. 重複 1-4 直到 一個 weight vector 裝滿 

#### Unpacking
* 從 weight vector 和 activation vector 中取出要相乘的 weight 和 activation (用 0000...00011 的 mask 和 vectors **做 &**，得到 lowest two bits)
* 取出來的 activation 往左移 2 bits，因為 lowest two bits 要放 weight
* 左移後的 activation 和 weight 做 OR (把 activation 和 weight 合在一起，變成 index)
* 得到 index，查 LUT
![image](https://hackmd.io/_uploads/r1d7BHNrJl.png)

### LUT16 Example
#### Steps
Given packed weight vector and activation vector:
Perform unpacking
* weights are reordered (so that we can skip the left shift in the unpacking process) (done offline)
* Apply a mask on the weight/activation vectors to extract two pairs of weights and activations at the same time (The mask used is slightly different from the mask mentioned above in the unpacking part, in order to reduce the number of operations needed to **unpack two pairs** of weights and activations.)
* use `or` to produce the index
    * indices are 0000 0001 0010 0011 ... 1111
* (right shift 2 bits to move the a/w or w/a pair to the correct position)
*  `shuffle` is the operation of looking up the table using the index produced in the previous step.
* This process produces 4, 8-bit dot products in vector registers 12,16,21,26.
* A **vectorized sum** of the 8-bit elements is performed across these registers. The final operation is a horizontal addition of these values.
![image](https://hackmd.io/_uploads/SJNQu74HJl.png)

#### Advantages of perform convolution using LUT
* Number of instructions to perform convolution is comparable to the FP32 baseline
* the latency of shuffle operation being lower than multiplication
* The number of values loaded in a register is larger => Fewer exchanges between cache and registers  
    * $R$: the vector register size
    * LUT: $\frac{R}{2}$
    * FP32:  $\frac{R}{32}$
* Scaling LUT-16 to larger bitwidths
    * The lookup table easily fits within a typical L1 cache on modern processor.
![image](https://hackmd.io/_uploads/H1E7C74Bkg.png)

### Experiments
#### Operator Profiling
*  On the x86 platform (Inteli79700k@3.6GHz)
*  DeepGEMM vs optimized INT8 kernels in the QNNPACK library
*  Compare performance on 3 classification models 
    *  x-axis: dimensions of GEMM computations of each layer
    *  y-axis: speedup over non-optimized Int8

![image](https://hackmd.io/_uploads/H1RsMwNB1l.png)

#### End-to-End Profiling
* End-to-end inference results for 6 CNNs tested with QNNPACK and DeepGEMM kernels
* All convolution layers are quantized to 8 bits for QNNPACK, 2 bits for DeepGEMM
* Speedup over QNNPACK
![image](https://hackmd.io/_uploads/ryMhld4Bkl.png)

* Minor Accuracy Degradation
![image](https://hackmd.io/_uploads/rkfqzvESyg.png)

#### Low-Level Kernel Profiling
* To see the bottleneck of the  DeepGEMM
* Operations for a single convolution layer are categorized into:
    1. activation quantization
    2. activation packing
    3. convolution with LUT 
        * includes unpacking, lookup, accumulation
    4. activation dequantization
*  Lut-Conv is the bottleneck (shown as red bar)
    *  Unpacking step within Lut-Conv consumes about 80% of overall execution time

![image](https://hackmd.io/_uploads/Bkl8rONSkx.png)

### Conclusion
* Introduce DeepGEMM
    a lookup based approach for CPUs that 
    * replaces the costly "multiply-accumulate arithmetic in dot product calculations" with simpler "indexing operations into preconstructed tables" for computing ultra low-precision layers in convolutional neural networks.
* Advantages of DeepGEMM
    * has lower latency and fewer memory accesses
    * Greater Flexibility
        * support both uniform and non-uniform quantization methods (because signed/unsigned integer/floating-point results can be stored into the lookup table)
* Implement vectorized DeepGEMM kernels for x86 platforms that outperform optimized 8-bit baselines by up to 1.74×
