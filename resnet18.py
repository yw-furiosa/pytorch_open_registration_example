import torch
import torchvision
import torch.utils.cpp_extension

# torch._logging.set_logs(all=10)

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


def custom_convolution_overrideable(
    input, weight, bias, stride, padding, dilation, trasposed, output_padding, groups
):
    w = input.shape[2]
    f = weight.shape[2]
    p = padding[0]
    s = stride[0]
    dim0 = input.shape[0]
    dim1 = weight.shape[0]
    dim2 = int((w - f + 2 * p) / s) + 1
    # print(f"output: ({dim0, dim1, dim2, dim2})")
    return torch.randn(dim0, dim1, dim2, dim2).to("npu:0")


lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
lib.impl("convolution_overrideable", custom_convolution_overrideable)


def backend(m, inputs):
    m.print_readable()
    return m


m = torchvision.models.resnet18().to("npu")
x = torch.randn(1, 3, 244, 244).to("npu:0")
m.eval()
m = torch.compile(m, backend=backend)

# pass convolution but fail because view operator is not implemented
m(x)
