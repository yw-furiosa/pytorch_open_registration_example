import torch
from utils.custom_device_mode import npu_module, enable_npu_device

# This file contains an example of how to create a custom device extension
# in PyTorch, through the dispatcher.
# It also shows what two possible user API's for custom devices look like. Either:
# (1) Expose your custom device as an object, device=my_device_obj
# (2) Allow users to directly use device strings: device="my_device"

# Running this file prints the following:

# (Correctly) unable to create tensor on device='bar'
# (Correctly) unable to create tensor on device='npu:2'
# Creating x on device 'npu:0'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# Creating y on device 'npu:0'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!

# Test START
# x.device=npu:0, x.is_cpu=False
# y.device=npu:0, y.is_cpu=False
# Calling z = x + y
# Custom aten::add.Tensor() called!
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# z.device=npu:0, z.is_cpu=False
# Calling z = z.to(device="cpu")
# Custom aten::_copy_from() called!
# z_cpu.device=cpu, z_cpu.is_cpu=True
# Calling z2 = z_cpu + z_cpu
# Test END

# Custom allocator's delete() called!
# Creating x on device 'npu:1'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# Creating y on device 'npu:1'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!

# Test START
# x.device=npu:0, x.is_cpu=False
# y.device=npu:0, y.is_cpu=False
# Calling z = x + y
# Custom aten::add.Tensor() called!
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# z.device=npu:0, z.is_cpu=False
# Calling z = z.to(device="cpu")
# Custom aten::_copy_from() called!
# z_cpu.device=cpu, z_cpu.is_cpu=True
# Calling z2 = z_cpu + z_cpu
# Test END

# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!

torch.utils.rename_privateuse1_backend('npu')
torch.utils.generate_methods_for_privateuse1_backend()

def test(x, y):
    print()
    print("Test START")
    # Check that our device is correct.
    print(f'x.device={x.device}, x.is_cpu={x.is_cpu}, x.is_npu={x.is_npu}')
    print(f'y.device={y.device}, y.is_cpu={y.is_cpu}, y.is_npu={y.is_npu}')

    # calls out custom add kernel, registered to the dispatcher
    print('Calling z = x + y')
    z = x + y
    print(f'z.device={z.device}, z.is_cpu={z.is_cpu}, z.is_npu={z.is_npu}')

    print('Calling z = z.to(device="cpu")')
    z_cpu = z.to(device='cpu')

    # Check that our cross-device copy correctly copied the data to cpu
    print(f'z_cpu.device={z_cpu.device}, z_cpu.is_cpu={z_cpu.is_cpu}, z_cpu.is_npu={z_cpu.is_npu}')

    # Confirm that calling the add kernel no longer invokes our custom kernel,
    # since we're using CPU t4ensors.
    print('Calling z2 = z_cpu + z_cpu')
    z2 = z_cpu + z_cpu
    print("Test END")
    print()

# Option 1: Use torch.register_privateuse1_backend("npu"), which will allow
# "npu" as a device string to work seamlessly with pytorch's API's.
# You may need a more recent nightly of PyTorch for this.

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='bar')
    exit("Error: you should not be able to make a tensor on an arbitrary 'bar' device.")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='bar'")

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='npu:2')
    exit("Error: the npu device only has two valid indices: npu:0 and npu:1")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='npu:2'")

print("Creating x on device 'npu:0'")
x1 = torch.ones(4, 4, device='npu:0')
print("Creating y on device 'npu:0'")
y1 = torch.ones(4, 4, device='npu:0')

test(x1, y1)


# Option 2: Directly expose a custom device object
# You can pass an optional index arg, specifying which device index to use.
npu_device1 = npu_module.custom_device(1)

print("Creating x on device 'npu:1'")
x2 = torch.ones(4, 4, device=npu_device1)
print("Creating y on device 'npu:1'")
y2 = torch.ones(4, 4, device=npu_device1)

# Option 3: Enable a TorchFunctionMode object in user land,
# that will convert `device="npu"` calls into our custom device objects automatically.
# Option 1 is strictly better here (in particular, printing a.device() will still
# print "privateuseone" instead of your custom device name). Mostly showing this option because:
# (a) Torch Function Modes have been around for longer, and the API in Option 1
#     is only available on a more recent nightly.
# (b) This is a cool example of how powerful torch_function and torch_dispatch modes can be!
# holder = enable_npu_device()
# del _holder

test(x2, y2)
