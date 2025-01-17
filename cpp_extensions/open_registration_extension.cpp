#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/DispatchKeySet.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>

#include <iostream>

// This file contains the heavy lifting to add a new C++ backend
// and integrate it directly into the PyTorch backend. It mainly involves:
//
// (1) Writing a custom allocator and registering it to pytorch
//     (see DummyCustomAllocator)
// (2) Writing a custom device guard, registering it to pytorch,
//     and using the device guard in kernels
//     (see DummyDeviceGuard)
// (3) Writing a custom aten::empty.memory_format function

// basic dummy add function (not used)
at::Tensor custom_add_Tensor(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << "Custom aten::add.Tensor() called!" << std::endl;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// =====================================
// ========= Custom Allocators =========
// =====================================

// PyTorch provides an API for registering custom allocators for your device.
// You can create one by inheriting from the at::Allocator class,
// and registering your allocator for the particular device type
// (PrivateUse1 for open registration devices)

// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator
{
  DummyCustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) const override
  {
    void *data = c10::alloc_cpu(nbytes);
    std::cout << "Custom allocator's allocate() called! Allocate " << nbytes << " at [" << data << "]" << std::endl;
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void *ptr)
  {
    if (!ptr)
    {
      return;
    }
    std::cout << "Custom allocator's delete() called! Free at [" << ptr << "]" << std::endl;
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override
  {
    return &ReportAndDelete;
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

// =====================================
// ============= Device Guards =========
// =====================================

// PyTorch has an API for registering device guards.
// Device guards can be used to set the current "active" device,
// and e.g. error if the user provides an invalid device index.
//
// If your device doesn't support indices (e.g. foo:0 vs. foo:1),
// then the guards probably aren't needed.
//
// You can use it by creating a DeviceGuard class, registering it
// in PyTorch, and invoking the device guard before any kernels are called.
// For a more full-featured example of a device guard,
// check out the code at c10/cuda/CUDAGuard.h

// Represents the current "active" device.
// The dummy device guard registered below is meant to show how a backend
// can integrate custom device guard with pytorch.
// For something like cuda this represents the current active cuda device,
// which is directly set using the cuda API calls cudaGetDevice/cudaSetDevice.
static uint16_t CURR_DEVICE = -1;

// Create and register a dummy device guard.
struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
  DummyDeviceGuardImpl() {}
  explicit DummyDeviceGuardImpl(c10::DeviceType t)
  {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }
  at::DeviceType type() const override
  {
    return at::DeviceType::PrivateUse1;
  }
  at::Device exchangeDevice(at::Device d) const override
  {
    TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    at::Device old_device = getDevice();
    if (old_device.index() != d.index())
    {
      // "set the active device"
      CURR_DEVICE = d.index();
    }
    return old_device;
  }
  at::Device getDevice() const override
  {
    return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
  }
  void setDevice(at::Device d) const override
  {
    TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    at::Device current_device = getDevice();
    if (current_device != d)
    {
      CURR_DEVICE = d.index();
    }
  }
  void uncheckedSetDevice(at::Device d) const noexcept override
  {
    auto current_device = getDevice();
    if (current_device != d)
    {
      CURR_DEVICE = d.index();
    }
  }
  at::Stream getStream(at::Device d) const noexcept override
  {
    // no-op
    return at::Stream(at::Stream::DEFAULT, d);
  }
  // NB: These do NOT set the current device
  at::Stream exchangeStream(at::Stream) const noexcept override
  {
    // no-op
    return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE));
  }
  at::DeviceIndex deviceCount() const noexcept override
  {
    // Hardcoding the number of "valid" devices here at 2.
    return 2;
  }

  // Event-related functions
  void record(
      void ** /*event*/,
      const at::Stream & /*stream*/,
      const at::DeviceIndex /*device_index*/,
      const c10::EventFlag /*flag*/) const override
  {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
  }
  void block(void * /*event*/, const at::Stream & /*stream*/) const override
  {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
  }
  bool queryEvent(void * /*event*/) const override
  {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
  }
  void destroyEvent(void * /*event*/, const at::DeviceIndex /*device_index*/)
      const noexcept override {}

  // Stream-related functions
  bool queryStream(const at::Stream & /*stream*/) const override
  {
    return true;
  }
  void synchronizeStream(const at::Stream & /*stream*/) const override
  {
    // Don't wait for anything.
  }
};

struct DummyGuard
{
  explicit DummyGuard() = delete;
  explicit DummyGuard(at::DeviceIndex device_index) : guard_(device_index) {}
  explicit DummyGuard(at::Device device) : guard_(device) {}
  DummyGuard(const DummyGuard &) = delete;
  DummyGuard &operator=(const DummyGuard &) = delete;
  DummyGuard(DummyGuard &&other) = delete;
  DummyGuard &operator=(DummyGuard &&other) = delete;

  void set_device(at::Device device)
  {
    guard_.set_device(device);
  }

  void reset_device(at::Device device)
  {
    guard_.reset_device(device);
  }

  void set_index(at::DeviceIndex device_index)
  {
    guard_.set_index(device_index);
  }

  at::Device original_device() const
  {
    return guard_.original_device();
  }

  at::Device current_device() const
  {
    return guard_.current_device();
  }

private:
  c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);

// =====================================
// ============= KERNELS ===============
// =====================================

// basic dummy empty function, so we can directly construct tensors on the custom device
// This dummy test device will just use the CPU allocator, and ignores pinned memory.
//
// Note: this kernel is very simple because our "custom device" just uses the normal TensorImpl object
// to store data under the hood.
// In PyTorch core today, both cpu and cuda are implemented with an ordinary TensorImpl class.
// Sometimes, backends prefer to subclass TensorImpl in order to store extra information.
// If this is the case, then this kernel is where you'll be responsible for creating and returning
// a fresh at::Tensor object, that properly stores a TensorImpl of your subclass.
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)
{
  const at::OptionalDeviceGuard device_guard(device);
  std::cout << "Custom aten::empty.memory_format() called!" << std::endl;
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor custom_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<c10::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
  const at::OptionalDeviceGuard device_guard(device);
  std::cout << "Custom aten::empty_strided() called!" << std::endl;
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype));
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking)
{
  // TODO: Fix semnatic of copy.
  // Src memory address is not the start of self, and dst memory address is also not the start of dst
  // src_memory_address = self.storage().data_ptr().get() + (self.itemsize() * self.storage_offset())
  // dst_memory_address = dst.storage().data_ptr().get() + (dst.itemsize() * dst.storage_offset())
  // n = calculate by size, stride, itemsize
  // memcpy(dst_memory_address, src_memory_address, n);

  // TODO: handle Meta tensor for tracing
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << "Custom aten::_copy_from() called! " << self.storage().data_ptr().get() << "[" << self.device() << "] -> " << dst.storage().data_ptr().get() << "[" << dst.device() << "] " << std::endl;
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  // Turn off this check because of meta tensor (meta tensor has no data so error in this check)
  // TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());
  // Revmoe this assertion because we implement type conversion
  // TORCH_CHECK(self.scalar_type() == dst.scalar_type());

  // Have to set storage_offset properly for indexing tensor
  // TODO: change to use tensor.set_ by implementing set_.source_Storage_storage_offset
  auto *dst_ = dst.unsafeGetTensorImpl();
  dst_->set_sizes_and_strides(dst.sizes(), dst.strides(), c10::make_optional(self.storage_offset()));

  // Type conversion
  if (self.scalar_type() != dst.scalar_type())
  {
    std::cout << "\tExecute custom type conversion" << std::endl;
    std::cout << "\t" << self.sizes() << " * " << self.scalar_type() << " [" << self.storage().nbytes() << "] -> " << dst.sizes() << " * " << dst.scalar_type() << " [" << dst.storage().nbytes() << "]" << std::endl;
    // refer https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/CopyKernel.cpp#L244
    auto iter = at::TensorIteratorConfig()
                    .add_output(dst)
                    .add_input(self)
                    .resize_outputs(false)
                    .check_all_same_dtype(false)
                    .check_all_same_device(false) // not sure
                    .build();

    AT_DISPATCH_ALL_TYPES(self.scalar_type(), "_copy_from", [&]
                          {
      using src_t = scalar_t;
      AT_DISPATCH_ALL_TYPES(dst.scalar_type(), "_copy_from", [&] {
        using dst_t = scalar_t;
        iter.for_each(
                      [](char** data, const int64_t* strides, int64_t size) {
                        auto src = reinterpret_cast<const src_t*>(data[1]);
                        auto dst = reinterpret_cast<dst_t*>(data[0]);
                        at::vec::convert(src, dst, size);
                      });
      }); });
    return dst;
  }

  // cpu -> npu
  if (self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1)
  {
    std::cout << "\tmemcpy from cpu to npu" << std::endl;
    std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  }
  // npu -> cpu
  else if (self.device().type() == c10::DeviceType::PrivateUse1 && dst.is_cpu())
  {
    std::cout << "\tcopy only data_ptr" << std::endl;
    at::DataPtr data_ptr = {self.storage().data_ptr().get(), self.storage().data_ptr().get_context(), nullptr, dst.device()};
    dst.storage().set_data_ptr_noswap(std::move(data_ptr));
  }
  // npu -> npu
  else if (self.device() == dst.device())
  {
    std::cout << "\tmemcpy from npu to npu " << self.storage().nbytes() << ", " << dst.storage().nbytes() << std::endl;
    if (self.storage().data_ptr().get() != dst.storage().data_ptr().get())
    {
      std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
    }
  }

  // Failed at assertion `result.storage().use_count() == 1`
  // auto p = const_cast<at::Tensor *>(&dst);
  // *p = at::Tensor(self);

  return dst;
}

at::Tensor custom__copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst)
{
  std::cout << "Custom aten::_copy_from_and_resize() called! " << self.storage().data_ptr().get() << "[" << self.device() << "] -> " << dst.storage().data_ptr().get() << "[" << dst.device() << "] " << std::endl;
  // The reason of handling this case here is because this case usually occurs in cpu_fallback,
  // and _copy_from_and_resize function is called only in cpu_fallback.
  if (self.storage().data_ptr().get() == dst.storage().data_ptr().get())
  {
    std::cout << "\tSkip _copy_from_and_resize() because they already pointing same data pointer" << std::endl;
    return dst;
  }

  // aten::abs operator generate output tensor as size 0 and resize it
  // Output tensor generated at first (dst) has size 0, but calculated output tensor from cpu_fallback (self) has valid size
  // So we have to resize dst tensor too before copy
  if (self.sizes() != dst.sizes() && dst.storage().nbytes() == 0)
  {
    std::cout << "\tRealloc dst storage " << dst.sizes() << " to " << self.sizes() << " and before copy" << std::endl;
    at::DataPtr data_ptr = dst.storage().allocator()->allocate(self.storage().nbytes());
    dst.storage().set_data_ptr_noswap(std::move(data_ptr));
    dst.storage().set_nbytes(self.storage().nbytes());
    // TODO: change to use tensor.set_ by implementing set_.source_Storage_storage_offset
    auto *dst_ = dst.unsafeGetTensorImpl();
    dst_->set_sizes_and_strides(self.sizes(), self.strides(), c10::make_optional(self.storage_offset()));
  }

  return custom__copy_from(self, dst, false);
}

// TODO: take only requried semenatics from at::native functions
at::Tensor custom_view(const at::Tensor &self, c10::IntArrayRef size)
{
  std::cout << "Custom aten::view() called!" << std::endl;
  return at::native::view(self, size);
}

// TODO: take only requried semenatics from at::native functions
at::Tensor custom__reshape_alias(const at::Tensor &self, c10::IntArrayRef size, c10::IntArrayRef stride)
{
  std::cout << "Custom aten::_reshape_alias() called!" << std::endl;
  return at::native::_reshape_alias(self, size, stride);
}

// TODO: take only requried semenatics from at::native functions
at::Tensor custom_as_strided(
    const at::Tensor &self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_)
{
  // copy implementation from build/aten/src/ATen/RegisterCPU.cpp wrapper_CPU__as_strided
  std::cout << "Custom aten::as_strided() called!" << std::endl;
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset_);
}

// TODO: take only requried semenatics from at::native functions
at::Tensor &custom_set__source_Storage(at::Tensor &result, c10::Storage src)
{
  std::cout << "Custom aten::set_.source_Storage() called!" << std::endl;
  int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
  c10::IntArrayRef stride = {};
  at::native::set_storage_cpu_(result, src, 0, new_size, stride);
  return result;
}

// TODO: take only requried semenatics from at::native functions
at::Tensor &custom_set__source_Storage_storage_offset(at::Tensor &result,
                                                      c10::Storage storage,
                                                      int64_t storage_offset,
                                                      c10::IntArrayRef size,
                                                      c10::IntArrayRef stride)
{
  std::cout << "Custom aten::set_.source_Sorage_storage_offset() called!" << std::endl;
  at::native::set_storage_cpu_(result, storage, storage_offset, size, stride);
  return result;
}

// not used yet
at::Tensor custom_index_Tensor_out(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices, at::Tensor &out)
{
  std::cout << "Custom aten::index.Tensor_out() called!" << std::endl;
  return out;
}

// not used yet
const at::Tensor &custom_resize_(const at::Tensor &self, at::IntArrayRef size,
                                 c10::optional<at::MemoryFormat> optional_memory_format)
{
  std::cout << "Custom aten::resize_() called!" << std::endl;
  return at::native::resize_(self, size, optional_memory_format);
}
// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.

// Refer XLA https://github.com/pytorch/xla/blob/e51d28b64c1757d62fb46a1263549140a2cb3ca2/torch_xla/csrc/aten_cpu_fallback.cpp#L18
// or Ascend https://github.com/Ascend/pytorch/blob/e70b14dcbb0ebf362cf2985a59113ee2247df956/torch_npu/csrc/aten/VariableFallbackKernel.cpp#L49
void custom_backend_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
  std::cout << "Custom backend_fallback() for " << op.operator_name() << " called!" << std::endl;
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_backend_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("empty_strided", &custom_empty_strided);
  // m.imp("fill_.Scalar", &custom_fill__Scalar);

  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
  m.impl("_copy_from", &custom__copy_from);

  m.impl("_reshape_alias", &custom__reshape_alias);
  m.impl("view", &custom_view);
  m.impl("as_strided", &custom_as_strided);
  // m.impl("index.Tensor_out", &custom_index_Tensor_out);

  m.impl("set_.source_Storage", &custom_set__source_Storage);
  m.impl("set_.source_Storage_storage_offset", &custom_set__source_Storage_storage_offset);

  // m.impl("_pin_memory", &custom__pin_memory);
  // m.impl("is_pinned", &custom_is_pinned);
  // m.impl("resize_", &custom_resize_);
}

// Make `furiosa` namespace and register two custom operators
at::Tensor furiosa_cpu_task(const at::Tensor &a, const at::Tensor &b)
{
  // dummy impl
  const at::OptionalDeviceGuard device_guard(at::device_of(a));
  std::cout << "Custom furiosa::cpu_task() called!" << std::endl;
  return at::empty(a.sizes(), a.options());
}

at::Tensor furiosa_npu_task(const at::Tensor &a, const at::Tensor &b)
{
  // dummy impl
  const at::OptionalDeviceGuard device_guard(at::device_of(b));
  std::cout << "Custom furiosa::npu_task() called!" << std::endl;
  return at::empty(b.sizes(), b.options());
}

TORCH_LIBRARY(furiosa, m)
{
  m.def("cpu_task(Tensor a, Tensor b) -> Tensor", &furiosa_cpu_task);
  m.def("npu_task(Tensor a, Tensor b) -> Tensor", &furiosa_npu_task);
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device(int idx)
{
  return c10::Device(c10::DeviceType::PrivateUse1, idx);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_device", &get_custom_device, "get custom device object");
}
