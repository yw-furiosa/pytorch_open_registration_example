import torch
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
import torch.utils.cpp_extension
import transformers


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


# schema: Tensor &index_out(const Tensor &self, const c10::List<c10::optional<Tensor>> &indices, Tensor &out);
def custom_index_tensor_out(self, indicies, out):
    print("Custom aten::index.Tensor_out() called!")
    # TODO: how this operator works?
    return out


lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
lib.impl("convolution_overrideable", custom_convolution_overrideable)
lib.impl("index.Tensor_out", custom_index_tensor_out)

decomposition_table = core_aten_decompositions()

backend_index = 0


def my_backend(gm, inputs):
    global backend_index
    print("===== original graph {} =====".format(backend_index))
    gm.print_readable()
    inputs_cloned = [i.clone().detach() for i in inputs]
    # inputs_cloned = [i.clone() if isinstance(i, torch.Tensor) else i for i in inputs]
    gm = make_fx(
        gm,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        decomposition_table=decomposition_table,
    )(*inputs_cloned)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    print("===== modified graph {} =====".format(backend_index))
    gm.print_readable()
    backend_index += 1
    return gm


m = transformers.GPT2Model.from_pretrained("gpt2").to("npu")
m = torch.compile(m, backend=my_backend)
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
inputs = tokenizer("test input here, long long message", return_tensors="pt").to("npu")
result = m(**inputs)
print("finish\n")
