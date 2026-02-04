#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // TODO：实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================

        using FuncT = RetT (*)(ArgsT...);
        CHECK(func_ptr_) << "KernelFunction::Call is invoked before registration";
        auto func = reinterpret_cast<FuncT>(func_ptr_);
        CHECK(func) << "KernelFunction::Call failed to reinterpret function pointer";
        return func(std::forward<ArgsT>(args)...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== 作业 ===================================
        // TODO：实现kernel注册机制
        // 功能描述：将kernel函数与设备类型、名称绑定
        // =================================== 作业 ===================================

        CHECK(!key_to_kernel_map_.contains(key)) << "Kernel already registered for " << key.second
                                                 << " on device " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};

namespace detail {
template <typename FuncT> class KernelRegistrar {
public:
    KernelRegistrar(const Dispatcher::KeyT &key, FuncT func) { Dispatcher::Instance().Register(key, func); }

    KernelRegistrar(DeviceType device, const char *name, FuncT func)
        : KernelRegistrar(Dispatcher::KeyT{device, name}, func) {}
};
} // namespace detail
} // namespace infini_train

#define INFTR_CONCAT_IMPL(a, b) a##b
#define INFTR_CONCAT(a, b) INFTR_CONCAT_IMPL(a, b)
#define INFTR_UNIQUE_NAME(base) INFTR_CONCAT(base, __COUNTER__)
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static ::infini_train::detail::KernelRegistrar<decltype(kernel_func)>                                              \
        INFTR_UNIQUE_NAME(_KernelRegistrar)(device, #kernel_name, kernel_func);
