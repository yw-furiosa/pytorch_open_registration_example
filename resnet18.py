import torch
import torchvision
import torch.utils.cpp_extension
import torch._dynamo.config
import logging
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True
torch._dynamo.config.set_loggers_level(10)

npu_module = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[
        "cpp_extensions/open_registration_extension.cpp",
    ],
    extra_include_paths=["cpp_extensions"],
    extra_cflags=["-g"],
    verbose=True,
)

torch.utils.rename_privateuse1_backend("npu")


# schema: Tensor convolution_overrideable(const Tensor &input, const Tensor &weight, const c10::optional<Tensor> &bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups);
def custom_convolution_overrideable(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    print("Custom aten::convolution_overrideable called!")
    origin_device = input.device
    if origin_device == torch.device("meta"):
        # for tracing
        # cannot copy tensor from meta tensor since they do not have any data
        print("\tcall convolution with random data for tracing")
        input = torch.randn(input.size())
        weight = torch.randn(weight.size())
        bias = torch.randn(bias.size()) if bias else None
    else:
        # for eager mode (act as cpu)
        print("\tcall convolution with copied data from input data to CPU")
        input = input.to("cpu")
        weight = weight.to("cpu")
        bias = bias.to("cpu") if bias else None

    return torch.ops.aten.convolution.default(
        input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    ).to(origin_device)


# schema: at::Tensor view(const at::Tensor& self, c10::IntArrayRef size)
def custom_view(self, size):
    print("Custom aten::view called!")
    if self.device == torch.device("meta"):
        # for tracing
        output = torch.randn(size).to(self.device)
    else:
        # for execution
        # The reason of using as_strided is that as_strided return a tensor which has same storage with input
        # It makes result can pass the assertion in ADInplaceOrView::view operator which check return tensor has same storage with input
        # That assertion is at torch/csrc/autograd/generated/VariableType_3.cpp::15467, "AT_ASSERT(self__storage_saved.value().is_alias_of(result.storage()));"
        stride = [1]
        for s in reversed(size):
            stride.insert(0, stride[0] * s)
        output = self.as_strided(size, stride[1:])
    return output


# schema: at::Tensor _reshape_alias(const at::Tensor &self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride)
def custom__reshape_alias(self, size, stride):
    print("Custom aten::_reshape_alias called!")
    return self.as_strided(size, stride)


lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
lib.impl("convolution_overrideable", custom_convolution_overrideable)
lib.impl("view", custom_view)
lib.impl("_reshape_alias", custom__reshape_alias)


def backend(m, inputs):
    # m = functionalize(m, remove="mutations_and_views")
    m = make_fx(m, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
    # m.print_readable()
    return m


m = torchvision.models.resnet18().to("npu")
x = torch.randn(1, 3, 244, 244).to("npu:0")
m.eval()
m = torch.compile(m, backend=backend)

y = m(x)
print(y.shape)
print("finish")
