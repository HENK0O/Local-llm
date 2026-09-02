#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LOCAL_LLM_ARM_NEON 1
#endif

namespace {

constexpr npy_intp kBlockValues = 32;
constexpr npy_intp kQ8BlockBytes = 34;
constexpr npy_intp kQ4BlockBytes = 18;
constexpr npy_intp kQ4PackedValues = 16;

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

struct MatmulContext {
    const char* blocks;
    const float* input;
    float* output;
    npy_intp rows;
    npy_intp blocks_per_row;
    npy_intp batches;
    std::size_t jobs;
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
        context->output[index] = result;
    }
}

PyObject* q8_matmul(PyObject*, PyObject* args) {
    PyObject* blocks_object;
    PyObject* input_object;
    if (!PyArg_ParseTuple(args, "OO:q8_matmul", &blocks_object, &input_object)) {
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

    const npy_intp total = batches * rows;
    const std::size_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
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
    };

    Py_BEGIN_ALLOW_THREADS
#ifdef __APPLE__
    dispatch_apply_f(context.jobs, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, run_job);
#else
    run_job(&context, 0);
#endif
    Py_END_ALLOW_THREADS

    Py_DECREF(blocks);
    Py_DECREF(input);
    return reinterpret_cast<PyObject*>(output);
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
            float dot = 0.0f;
#ifdef __clang__
#pragma clang loop vectorize(enable)
#endif
            for (npy_intp value = 0; value < kQ4PackedValues; ++value) {
                const std::uint8_t byte = quantized[value];
                dot += static_cast<float>(static_cast<int>(byte & 0x0fu) - 8) *
                       input_block[value];
                dot += static_cast<float>(static_cast<int>(byte >> 4) - 8) *
                       input_block[value + kQ4PackedValues];
            }
            result += scale * dot;
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
    const std::size_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
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
    dispatch_apply_f(context.jobs, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
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
