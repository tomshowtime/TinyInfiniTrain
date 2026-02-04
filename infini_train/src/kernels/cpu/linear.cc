#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace {
std::string DimsToString(const std::vector<int64_t> &dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<int64_t> PadDimsLeft(const std::vector<int64_t> &dims, size_t target_rank) {
    if (dims.size() >= target_rank) {
        return dims;
    }
    std::vector<int64_t> padded(target_rank - dims.size(), 1);
    padded.insert(padded.end(), dims.begin(), dims.end());
    return padded;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }
    return strides;
}

std::vector<int64_t> BroadcastBatchDims(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs) {
    CHECK_EQ(lhs.size(), rhs.size());
    CHECK_GE(lhs.size(), 2);
    std::vector<int64_t> batch_dims(lhs.size() - 2);
    for (size_t i = 0; i < batch_dims.size(); ++i) {
        const int64_t a = lhs[i];
        const int64_t b = rhs[i];
        if (a == b) {
            batch_dims[i] = a;
        } else if (a == 1) {
            batch_dims[i] = b;
        } else if (b == 1) {
            batch_dims[i] = a;
        } else {
            LOG(FATAL) << "Incompatible batch dimensions: " << a << " vs " << b << " at axis " << i;
        }
    }
    return batch_dims;
}
} // namespace

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================

    CHECK(input);
    CHECK(other);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(other->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_GE(input->Dims().size(), 2);
    CHECK_GE(other->Dims().size(), 2);

    const size_t target_rank = std::max(input->Dims().size(), other->Dims().size());
    auto a_dims = PadDimsLeft(input->Dims(), target_rank);
    auto b_dims = PadDimsLeft(other->Dims(), target_rank);

    const int64_t m = a_dims[target_rank - 2];
    const int64_t k = a_dims[target_rank - 1];
    const int64_t k_rhs = b_dims[target_rank - 2];
    const int64_t n = b_dims[target_rank - 1];
    CHECK_EQ(k, k_rhs) << "Matmul dimension mismatch, lhs K=" << k << ", rhs K=" << k_rhs;

    auto batch_dims = BroadcastBatchDims(a_dims, b_dims);
    const int64_t batch_count
        = batch_dims.empty() ? 1 : std::accumulate(batch_dims.begin(), batch_dims.end(), int64_t{1},
                                                   std::multiplies<int64_t>());

    std::vector<int64_t> output_dims = batch_dims;
    output_dims.push_back(m);
    output_dims.push_back(n);
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    float *out_ptr = static_cast<float *>(output->DataPtr());
    const float *a_ptr = static_cast<const float *>(input->DataPtr());
    const float *b_ptr = static_cast<const float *>(other->DataPtr());

    std::vector<int64_t> coords(batch_dims.size(), 0);
    const int64_t out_batch_stride = m * n;

    for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        int64_t tmp = batch_idx;
        for (int64_t axis = static_cast<int64_t>(batch_dims.size()) - 1; axis >= 0; --axis) {
            const int64_t dim = batch_dims[axis];
            coords[axis] = dim == 0 ? 0 : tmp % dim;
            tmp = dim == 0 ? 0 : tmp / dim;
        }

        int64_t a_offset = 0;
        int64_t b_offset = 0;
        for (size_t axis = 0; axis < batch_dims.size(); ++axis) {
            if (a_dims[axis] != 1) {
                a_offset += coords[axis] * a_strides[axis];
            }
            if (b_dims[axis] != 1) {
                b_offset += coords[axis] * b_strides[axis];
            }
        }

        const float *a_batch = a_ptr + a_offset;
        const float *b_batch = b_ptr + b_offset;
        float *out_batch = out_ptr + batch_idx * out_batch_stride;

        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int64_t kk = 0; kk < k; ++kk) {
                    acc += a_batch[i * k + kk] * b_batch[kk * n + j];
                }
                out_batch[i * n + j] = acc;
            }
        }
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================

    CHECK(input);
    CHECK(other);
    CHECK(grad_output);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(other->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32));

    const size_t target_rank = std::max(input->Dims().size(), other->Dims().size());
    auto a_dims = PadDimsLeft(input->Dims(), target_rank);
    auto b_dims = PadDimsLeft(other->Dims(), target_rank);
    const int64_t m = a_dims[target_rank - 2];
    const int64_t k = a_dims[target_rank - 1];
    const int64_t n = b_dims[target_rank - 1];
    CHECK_EQ(a_dims[target_rank - 1], b_dims[target_rank - 2]);

    auto batch_dims = BroadcastBatchDims(a_dims, b_dims);
    std::vector<int64_t> expected_output_dims = batch_dims;
    expected_output_dims.push_back(m);
    expected_output_dims.push_back(n);
    CHECK(expected_output_dims == grad_output->Dims())
        << "Grad output dims mismatch, expected " << DimsToString(expected_output_dims) << " but got "
        << DimsToString(grad_output->Dims());

    const int64_t batch_count
        = batch_dims.empty() ? 1 : std::accumulate(batch_dims.begin(), batch_dims.end(), int64_t{1},
                                                   std::multiplies<int64_t>());

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    const auto a_strides = ComputeStrides(a_dims);
    const auto b_strides = ComputeStrides(b_dims);
    const auto grad_output_stride = m * n;

    const float *a_ptr = static_cast<const float *>(input->DataPtr());
    const float *b_ptr = static_cast<const float *>(other->DataPtr());
    const float *grad_out_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_a_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_b_ptr = static_cast<float *>(grad_other->DataPtr());

    std::vector<int64_t> coords(batch_dims.size(), 0);
    for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        int64_t tmp = batch_idx;
        for (int64_t axis = static_cast<int64_t>(batch_dims.size()) - 1; axis >= 0; --axis) {
            const int64_t dim = batch_dims[axis];
            coords[axis] = dim == 0 ? 0 : tmp % dim;
            tmp = dim == 0 ? 0 : tmp / dim;
        }

        int64_t a_offset = 0;
        int64_t b_offset = 0;
        for (size_t axis = 0; axis < batch_dims.size(); ++axis) {
            if (a_dims[axis] != 1) {
                a_offset += coords[axis] * a_strides[axis];
            }
            if (b_dims[axis] != 1) {
                b_offset += coords[axis] * b_strides[axis];
            }
        }

        const float *a_batch = a_ptr + a_offset;
        const float *b_batch = b_ptr + b_offset;
        const float *grad_out_batch = grad_out_ptr + batch_idx * grad_output_stride;
        float *grad_a_batch = grad_a_ptr + a_offset;
        float *grad_b_batch = grad_b_ptr + b_offset;

        // dL/dA = dL/dC * B^T
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t kk = 0; kk < k; ++kk) {
                float acc = 0.0f;
                for (int64_t j = 0; j < n; ++j) {
                    acc += grad_out_batch[i * n + j] * b_batch[kk * n + j];
                }
                grad_a_batch[i * k + kk] += acc;
            }
        }

        // dL/dB = A^T * dL/dC
        for (int64_t kk = 0; kk < k; ++kk) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int64_t i = 0; i < m; ++i) {
                    acc += a_batch[i * k + kk] * grad_out_batch[i * n + j];
                }
                grad_b_batch[kk * n + j] += acc;
            }
        }
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
