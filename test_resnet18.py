import torch
import torchvision
import torch.utils.cpp_extension
import torch._dynamo.config
import torch.utils.data
from torch._dynamo.backends.common import aot_autograd

# import furiosa_torch_impl

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
        print("\tcall convolution with random tensor for tracing")
        input = torch.randn(input.size())
        weight = torch.randn(weight.size())
        bias = torch.randn(bias.size()) if bias else None
    else:
        # for eager mode (act as execute in cpu)
        print("\tcall convolution with input tensor for execution")
        input = input.to("cpu")
        weight = weight.to("cpu")
        bias = bias.to("cpu") if bias else None

    return torch.ops.aten.convolution.default(
        input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    ).to(origin_device)


lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
lib.impl("convolution_overrideable", custom_convolution_overrideable)


# backend = furiosa_torch_impl.Warboy(num_calib=1)
def backend(gm, inputs):
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(gm, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
    return gm


# backend = aot_autograd(fw_compiler=backend)

x = torch.randn(1, 3, 244, 244)
m = torchvision.models.resnet18(pretrained=True)
m.eval()

y_ref = m(x)
print("all done in cpu")


# x = x.to("npu")
# m = m.to("npu")
print("load model to npu")

m = torch.compile(m, backend=backend, dynamic=False)

print("start execution")
y = m(x)
assert all(y_ref[0][i].item() == y[0][i].item() for i in range(1000))
