#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LOCAL_LLM_ARM_NEON 1
#endif

namespace {

constexpr npy_intp kBlockValues = 32;
constexpr npy_intp kQ8BlockBytes = 34;
constexpr npy_intp kQ4BlockBytes = 18;
[[maybe_unused]] constexpr npy_intp kQ4PackedValues = 16;

std::size_t requested_threads(npy_intp batches) {
    static const std::size_t detected = std::max(1u, std::thread::hardware_concurrency());
    static const long configured_threads = []() {
        const char* configured = std::getenv("LOCAL_LLM_THREADS");
        if (configured == nullptr || *configured == '\0') {
            return 0L;
        }
        char* end = nullptr;
        const long value = std::strtol(configured, &end, 10);
        return end != configured && *end == '\0' && value > 0 ? value : 0L;
    }();
    if (configured_threads > 0) {
        return static_cast<std::size_t>(configured_threads);
    }
    if (batches == 1) {
#ifdef __APPLE__
        static const std::uint32_t performance_cores = []() {
            std::uint32_t cores = 0;
            std::size_t size = sizeof(cores);
            if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size,
                             nullptr, 0) != 0) {
                return std::uint32_t{0};
            }
            return cores;
        }();
        if (performance_cores > 0) {
            return performance_cores;
        }
#endif
    }
    return detected;
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float dot_q8_f32(const std::int8_t* quantized, const float* input) {
#ifdef LOCAL_LLM_ARM_NEON
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    for (npy_intp value = 0; value < kBlockValues; value += 8) {
        const int8x8_t packed = vld1_s8(quantized + value);
        const int16x8_t wide16 = vmovl_s8(packed);
        const float32x4_t low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide16)));
        const float32x4_t high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide16)));
        accumulator = vmlaq_f32(accumulator, low, vld1q_f32(input + value));
        accumulator = vmlaq_f32(accumulator, high, vld1q_f32(input + value + 4));
    }
#if defined(__aarch64__)
    return vaddvq_f32(accumulator);
#else
    const float32x2_t halves = vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
    const float32x2_t sum = vpadd_f32(halves, halves);
    return vget_lane_f32(sum, 0);
#endif
#else
    float dot = 0.0f;
    for (npy_intp value = 0; value < kBlockValues; ++value) {
        dot += static_cast<float>(quantized[value]) * input[value];
    }
    return dot;
#endif
}

float dot_q4_f32(const std::uint8_t* quantized, const float* input) {
#ifdef LOCAL_LLM_ARM_NEON
    const uint8x16_t packed = vld1q_u8(quantized);
    const uint8x16_t offset = vdupq_n_u8(8);
    const int8x16_t low = vreinterpretq_s8_u8(
        vsubq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)), offset));
    const int8x16_t high = vreinterpretq_s8_u8(
        vsubq_u8(vshrq_n_u8(packed, 4), offset));
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    const int16x8_t low_first = vmovl_s8(vget_low_s8(low));
    const int16x8_t low_second = vmovl_s8(vget_high_s8(low));
    const int16x8_t high_first = vmovl_s8(vget_low_s8(high));
    const int16x8_t high_second = vmovl_s8(vget_high_s8(high));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_first))),
        vld1q_f32(input));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_first))),
        vld1q_f32(input + 4));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(low_second))),
        vld1q_f32(input + 8));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low_second))),
        vld1q_f32(input + 12));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_first))),
        vld1q_f32(input + 16));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_first))),
        vld1q_f32(input + 20));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high_second))),
        vld1q_f32(input + 24));
    accumulator = vmlaq_f32(
        accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high_second))),
        vld1q_f32(input + 28));
#if defined(__aarch64__)
    return vaddvq_f32(accumulator);
#else
    const float32x2_t halves = vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
    const float32x2_t sum = vpadd_f32(halves, halves);
    return vget_lane_f32(sum, 0);
#endif
#else
    float dot = 0.0f;
    for (npy_intp value = 0; value < kQ4PackedValues; ++value) {
        const std::uint8_t byte = quantized[value];
        dot += static_cast<float>(static_cast<int>(byte & 0x0fu) - 8) * input[value];
        dot += static_cast<float>(static_cast<int>(byte >> 4) - 8) *
               input[value + kQ4PackedValues];
    }
    return dot;
#endif
}

struct MatmulContext {
    const char* blocks;
    const float* input;
    float* output;
    npy_intp rows;
    npy_intp blocks_per_row;
    npy_intp batches;
    std::size_t jobs;
    const float* add;
};

void run_job(void* raw_context, std::size_t job) {
    auto* context = static_cast<MatmulContext*>(raw_context);
    const npy_intp total = context->batches * context->rows;
    const npy_intp begin = total * static_cast<npy_intp>(job) /
                           static_cast<npy_intp>(context->jobs);
    const npy_intp end = total * static_cast<npy_intp>(job + 1) /
                         static_cast<npy_intp>(context->jobs);
    const npy_intp input_size = context->blocks_per_row * kBlockValues;
    const npy_intp row_bytes = context->blocks_per_row * kQ8BlockBytes;

    for (npy_intp index = begin; index < end; ++index) {
        const npy_intp batch = index / context->rows;
        const npy_intp row = index % context->rows;
        const float* x = context->input + batch * input_size;
        const char* row_blocks = context->blocks + row * row_bytes;
        float result = 0.0f;
        for (npy_intp block = 0; block < context->blocks_per_row; ++block) {
            const char* packed = row_blocks + block * kQ8BlockBytes;
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, packed, sizeof(scale_bits));
            const float scale = half_to_float(scale_bits);
            const auto* quantized = reinterpret_cast<const std::int8_t*>(packed + 2);
            const float* input_block = x + block * kBlockValues;
            result += scale * dot_q8_f32(quantized, input_block);
        }
        context->output[index] = result + (context->add == nullptr ? 0.0f : context->add[index]);
    }
}

PyObject* q8_matmul(PyObject*, PyObject* args) {
    PyObject* blocks_object;
    PyObject* input_object;
    PyObject* add_object = Py_None;
    if (!PyArg_ParseTuple(args, "OO|O:q8_matmul", &blocks_object, &input_object,
                          &add_object)) {
        return nullptr;
    }

    auto* blocks = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        blocks_object, nullptr, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr));
    auto* input = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        input_object, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (blocks == nullptr || input == nullptr) {
        Py_XDECREF(blocks);
        Py_XDECREF(input);
        return nullptr;
    }
    if (PyArray_ITEMSIZE(blocks) != kQ8BlockBytes) {
        PyErr_SetString(PyExc_ValueError, "Q8 blocks must contain 34 bytes per block");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }
    if (PyArray_NDIM(input) < 1) {
        PyErr_SetString(PyExc_ValueError, "Q8 input must have at least one dimension");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }

    const npy_intp rows = PyArray_DIM(blocks, 0);
    const npy_intp blocks_per_row = PyArray_DIM(blocks, 1);
    const int input_ndim = PyArray_NDIM(input);
    if (PyArray_DIM(input, input_ndim - 1) != blocks_per_row * kBlockValues) {
        PyErr_SetString(PyExc_ValueError, "Q8 input size does not match packed matrix");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }

    npy_intp batches = 1;
    for (int dimension = 0; dimension < input_ndim - 1; ++dimension) {
        batches *= PyArray_DIM(input, dimension);
    }
    npy_intp output_dimensions[NPY_MAXDIMS];
    for (int dimension = 0; dimension < input_ndim - 1; ++dimension) {
        output_dimensions[dimension] = PyArray_DIM(input, dimension);
    }
    output_dimensions[input_ndim - 1] = rows;
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(input_ndim, output_dimensions, NPY_FLOAT32));
    if (output == nullptr) {
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }
    PyArrayObject* add = nullptr;
    if (add_object != Py_None) {
        add = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            add_object, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
        bool valid = add != nullptr && PyArray_NDIM(add) == input_ndim;
        for (int dimension = 0; valid && dimension < input_ndim; ++dimension) {
            valid = PyArray_DIM(add, dimension) == output_dimensions[dimension];
        }
        if (!valid) {
            PyErr_SetString(PyExc_ValueError, "Q8 residual shape does not match output");
            Py_XDECREF(add);
            Py_DECREF(output);
            Py_DECREF(blocks);
            Py_DECREF(input);
            return nullptr;
        }
    }

    const npy_intp total = batches * rows;
    const std::size_t hardware_threads = requested_threads(batches);
#ifdef __APPLE__
    const std::size_t jobs = std::min<std::size_t>(
        hardware_threads, static_cast<std::size_t>(std::max<npy_intp>(1, total)));
#else
    const std::size_t jobs = 1;
#endif
    MatmulContext context{
        static_cast<const char*>(PyArray_DATA(blocks)),
        static_cast<const float*>(PyArray_DATA(input)),
        static_cast<float*>(PyArray_DATA(output)),
        rows,
        blocks_per_row,
        batches,
        jobs,
        add == nullptr ? nullptr : static_cast<const float*>(PyArray_DATA(add)),
    };

    Py_BEGIN_ALLOW_THREADS
#ifdef __APPLE__
    dispatch_apply_f(context.jobs, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     &context, run_job);
#else
    run_job(&context, 0);
#endif
    Py_END_ALLOW_THREADS

    Py_DECREF(blocks);
    Py_DECREF(input);
    Py_XDECREF(add);
    return reinterpret_cast<PyObject*>(output);
}

struct Q8PairContext {
    const char* first_blocks;
    const char* second_blocks;
    const float* input;
    float* first_output;
    float* second_output;
    npy_intp rows;
    npy_intp blocks_per_row;
    npy_intp batches;
    std::size_t jobs;
    bool swiglu;
};

void run_q8_pair_job(void* raw_context, std::size_t job) {
    auto* context = static_cast<Q8PairContext*>(raw_context);
    const npy_intp total = context->batches * context->rows;
    const npy_intp begin = total * static_cast<npy_intp>(job) /
                           static_cast<npy_intp>(context->jobs);
    const npy_intp end = total * static_cast<npy_intp>(job + 1) /
                         static_cast<npy_intp>(context->jobs);
    const npy_intp input_size = context->blocks_per_row * kBlockValues;
    const npy_intp row_bytes = context->blocks_per_row * kQ8BlockBytes;

    for (npy_intp index = begin; index < end; ++index) {
        const npy_intp batch = index / context->rows;
        const npy_intp row = index % context->rows;
        const float* x = context->input + batch * input_size;
        const char* first_row = context->first_blocks + row * row_bytes;
        const char* second_row = context->second_blocks + row * row_bytes;
        float first_result = 0.0f;
        float second_result = 0.0f;
        for (npy_intp block = 0; block < context->blocks_per_row; ++block) {
            const char* first = first_row + block * kQ8BlockBytes;
            const char* second = second_row + block * kQ8BlockBytes;
            std::uint16_t first_scale_bits;
            std::uint16_t second_scale_bits;
            std::memcpy(&first_scale_bits, first, sizeof(first_scale_bits));
            std::memcpy(&second_scale_bits, second, sizeof(second_scale_bits));
            const float* input_block = x + block * kBlockValues;
            first_result += half_to_float(first_scale_bits) * dot_q8_f32(
                reinterpret_cast<const std::int8_t*>(first + 2), input_block);
            second_result += half_to_float(second_scale_bits) * dot_q8_f32(
                reinterpret_cast<const std::int8_t*>(second + 2), input_block);
        }
        if (context->swiglu) {
            const float exponential = std::exp(
                first_result >= 0.0f ? -first_result : first_result);
            const float activated = first_result >= 0.0f
                ? first_result / (1.0f + exponential)
                : first_result * exponential / (1.0f + exponential);
            context->first_output[index] = activated * second_result;
        } else {
            context->first_output[index] = first_result;
            context->second_output[index] = second_result;
        }
    }
}

PyObject* q8_matmul_pair(PyObject*, PyObject* args) {
    PyObject* first_object;
    PyObject* second_object;
    PyObject* input_object;
    int swiglu = 0;
    if (!PyArg_ParseTuple(args, "OOO|p:q8_matmul_pair", &first_object, &second_object,
                          &input_object, &swiglu)) {
        return nullptr;
    }
    auto* first = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        first_object, nullptr, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr));
    auto* second = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        second_object, nullptr, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr));
    auto* input = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        input_object, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (first == nullptr || second == nullptr || input == nullptr) {
        Py_XDECREF(first);
        Py_XDECREF(second);
        Py_XDECREF(input);
        return nullptr;
    }
    if (PyArray_ITEMSIZE(first) != kQ8BlockBytes ||
        PyArray_ITEMSIZE(second) != kQ8BlockBytes ||
        PyArray_DIM(first, 0) != PyArray_DIM(second, 0) ||
        PyArray_DIM(first, 1) != PyArray_DIM(second, 1)) {
        PyErr_SetString(PyExc_ValueError, "paired Q8 matrices must have identical packed shapes");
        Py_DECREF(first);
        Py_DECREF(second);
        Py_DECREF(input);
        return nullptr;
    }
    if (PyArray_NDIM(input) < 1) {
        PyErr_SetString(PyExc_ValueError, "Q8 input must have at least one dimension");
        Py_DECREF(first);
        Py_DECREF(second);
        Py_DECREF(input);
        return nullptr;
    }
    const npy_intp rows = PyArray_DIM(first, 0);
    const npy_intp blocks_per_row = PyArray_DIM(first, 1);
    const int input_ndim = PyArray_NDIM(input);
    if (PyArray_DIM(input, input_ndim - 1) != blocks_per_row * kBlockValues) {
        PyErr_SetString(PyExc_ValueError, "Q8 input size does not match paired matrices");
        Py_DECREF(first);
        Py_DECREF(second);
        Py_DECREF(input);
        return nullptr;
    }

    npy_intp batches = 1;
    npy_intp output_dimensions[NPY_MAXDIMS];
    for (int dimension = 0; dimension < input_ndim - 1; ++dimension) {
        batches *= PyArray_DIM(input, dimension);
        output_dimensions[dimension] = PyArray_DIM(input, dimension);
    }
    output_dimensions[input_ndim - 1] = rows;
    auto* first_output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(input_ndim, output_dimensions, NPY_FLOAT32));
    auto* second_output = swiglu ? nullptr : reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(input_ndim, output_dimensions, NPY_FLOAT32));
    if (first_output == nullptr || (!swiglu && second_output == nullptr)) {
        Py_XDECREF(first_output);
        Py_XDECREF(second_output);
        Py_DECREF(first);
        Py_DECREF(second);
        Py_DECREF(input);
        return nullptr;
    }

    const npy_intp total = batches * rows;
    const std::size_t hardware_threads = requested_threads(batches);
#ifdef __APPLE__
    const std::size_t jobs = std::min<std::size_t>(
        hardware_threads, static_cast<std::size_t>(std::max<npy_intp>(1, total)));
#else
    const std::size_t jobs = 1;
#endif
    Q8PairContext context{
        static_cast<const char*>(PyArray_DATA(first)),
        static_cast<const char*>(PyArray_DATA(second)),
        static_cast<const float*>(PyArray_DATA(input)),
        static_cast<float*>(PyArray_DATA(first_output)),
        second_output == nullptr ? nullptr : static_cast<float*>(PyArray_DATA(second_output)),
        rows,
        blocks_per_row,
        batches,
        jobs,
        swiglu != 0,
    };
    Py_BEGIN_ALLOW_THREADS
#ifdef __APPLE__
    dispatch_apply_f(context.jobs, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     &context, run_q8_pair_job);
#else
    run_q8_pair_job(&context, 0);
#endif
    Py_END_ALLOW_THREADS

    Py_DECREF(first);
    Py_DECREF(second);
    Py_DECREF(input);
    if (swiglu) {
        return reinterpret_cast<PyObject*>(first_output);
    }
    auto* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(first_output);
        Py_DECREF(second_output);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, reinterpret_cast<PyObject*>(first_output));
    PyTuple_SET_ITEM(result, 1, reinterpret_cast<PyObject*>(second_output));
    return result;
}

struct Q4MatmulContext {
    const char* blocks;
    const float* input;
    float* output;
    npy_intp rows;
    npy_intp blocks_per_row;
    npy_intp batches;
    std::size_t jobs;
};

void run_q4_job(void* raw_context, std::size_t job) {
    auto* context = static_cast<Q4MatmulContext*>(raw_context);
    const npy_intp total = context->batches * context->rows;
    const npy_intp begin = total * static_cast<npy_intp>(job) /
                           static_cast<npy_intp>(context->jobs);
    const npy_intp end = total * static_cast<npy_intp>(job + 1) /
                         static_cast<npy_intp>(context->jobs);
    const npy_intp input_size = context->blocks_per_row * kBlockValues;
    const npy_intp row_bytes = context->blocks_per_row * kQ4BlockBytes;

    for (npy_intp index = begin; index < end; ++index) {
        const npy_intp batch = index / context->rows;
        const npy_intp row = index % context->rows;
        const float* x = context->input + batch * input_size;
        const char* row_blocks = context->blocks + row * row_bytes;
        float result = 0.0f;
        for (npy_intp block = 0; block < context->blocks_per_row; ++block) {
            const char* packed = row_blocks + block * kQ4BlockBytes;
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, packed, sizeof(scale_bits));
            const float scale = half_to_float(scale_bits);
            const auto* quantized = reinterpret_cast<const std::uint8_t*>(packed + 2);
            const float* input_block = x + block * kBlockValues;
            result += scale * dot_q4_f32(quantized, input_block);
        }
        context->output[index] = result;
    }
}

PyObject* q4_matmul(PyObject*, PyObject* args) {
    PyObject* blocks_object;
    PyObject* input_object;
    if (!PyArg_ParseTuple(args, "OO:q4_matmul", &blocks_object, &input_object)) {
        return nullptr;
    }

    auto* blocks = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        blocks_object, nullptr, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr));
    auto* input = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        input_object, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (blocks == nullptr || input == nullptr) {
        Py_XDECREF(blocks);
        Py_XDECREF(input);
        return nullptr;
    }
    if (PyArray_ITEMSIZE(blocks) != kQ4BlockBytes) {
        PyErr_SetString(PyExc_ValueError, "Q4 blocks must contain 18 bytes per block");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }
    if (PyArray_NDIM(input) < 1) {
        PyErr_SetString(PyExc_ValueError, "Q4 input must have at least one dimension");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }

    const npy_intp rows = PyArray_DIM(blocks, 0);
    const npy_intp blocks_per_row = PyArray_DIM(blocks, 1);
    const int input_ndim = PyArray_NDIM(input);
    if (PyArray_DIM(input, input_ndim - 1) != blocks_per_row * kBlockValues) {
        PyErr_SetString(PyExc_ValueError, "Q4 input size does not match packed matrix");
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }

    npy_intp batches = 1;
    for (int dimension = 0; dimension < input_ndim - 1; ++dimension) {
        batches *= PyArray_DIM(input, dimension);
    }
    npy_intp output_dimensions[NPY_MAXDIMS];
    for (int dimension = 0; dimension < input_ndim - 1; ++dimension) {
        output_dimensions[dimension] = PyArray_DIM(input, dimension);
    }
    output_dimensions[input_ndim - 1] = rows;
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(input_ndim, output_dimensions, NPY_FLOAT32));
    if (output == nullptr) {
        Py_DECREF(blocks);
        Py_DECREF(input);
        return nullptr;
    }

    const npy_intp total = batches * rows;
    const std::size_t hardware_threads = requested_threads(batches);
#ifdef __APPLE__
    const std::size_t jobs = std::min<std::size_t>(
        hardware_threads, static_cast<std::size_t>(std::max<npy_intp>(1, total)));
#else
    const std::size_t jobs = 1;
#endif
    Q4MatmulContext context{
        static_cast<const char*>(PyArray_DATA(blocks)),
        static_cast<const float*>(PyArray_DATA(input)),
        static_cast<float*>(PyArray_DATA(output)),
        rows,
        blocks_per_row,
        batches,
        jobs,
    };

    Py_BEGIN_ALLOW_THREADS
#ifdef __APPLE__
    dispatch_apply_f(context.jobs, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     &context, run_q4_job);
#else
    run_q4_job(&context, 0);
#endif
    Py_END_ALLOW_THREADS

    Py_DECREF(blocks);
    Py_DECREF(input);
    return reinterpret_cast<PyObject*>(output);
}

PyMethodDef methods[] = {
    {"q4_matmul", q4_matmul, METH_VARARGS, "Multiply packed GGML Q4_0 rows by float32 vectors."},
    {"q8_matmul", q8_matmul, METH_VARARGS, "Multiply packed GGML Q8_0 rows by float32 vectors."},
    {"q8_matmul_pair", q8_matmul_pair, METH_VARARGS, "Multiply two same-shaped Q8_0 matrices in one dispatch."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Optional native kernels for local-llm.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() {
    import_array();
    return PyModule_Create(&module);
}
