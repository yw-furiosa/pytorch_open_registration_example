import torch
import torch.utils.cpp_extension

npu_module = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[
        "cpp_extensions/open_registration_extension.cpp",
    ],
    extra_include_paths=["cpp_extensions"],
    extra_cflags=["-g"],
    verbose=True,
)

torch.utils.rename_privateuse1_backend('npu')
torch.utils.generate_methods_for_privateuse1_backend()

x = torch.randn(4, 4).to('npu:0')
y = torch.randn(4, 4).to('npu:0')

# furiosa::cpu_task -> furiosa_cpu_task
z_cpu = torch.ops.furiosa.cpu_task(x, y)
# furiosa::npu_task -> furiosa_npu_task
z_npu = torch.ops.furiosa.npu_task(x, y)
