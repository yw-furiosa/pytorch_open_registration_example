import torch
import torch.utils.cpp_extension

# load custom cpp extension
npu_module = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[
        "cpp_extensions/open_registration_extension.cpp",
    ],
    extra_include_paths=["cpp_extensions"],
    extra_cflags=["-g"],
    verbose=True,
)

# privateuse1 backend를 npu로 rename 후 *.npu(), *.is_npu method 생성
torch.utils.rename_privateuse1_backend('npu')
torch.utils.generate_methods_for_privateuse1_backend()

# [4, 4] shape의 tensor를 npu:0 device에 할당
# 함수 호출 순서: aten::empty.memory_format -> Allocator::alloc -> aten::_copy_from
x = torch.ones(4, 4).to('npu:0')
print('----------------')
# 함수 호출 순서: aten::empty_strided -> Allocator::alloc -> aten::_copy_from
y = torch.tensor([[0,1,2,3] for _ in range(4)], device='npu:0')

# call aten::add.Tensor -> custom_add_Tensor
z = x + y # z.is_cpu == False, z.is_npu == True
# print(z) # this line return error because there is no implementation of aten::view in PrivateUse1 backend

# call aten::_copy_from -> custom__copy_from
z = z.to('cpu') # z.is_cpu == True, z.is_npu == False
# print(z) # print dummy value because custom_add_Tensor return dummy tensor

# call aten::add.Tensor -> add.Tensor implementation of CPU backend
z_cpu = x.to('cpu') + y.to('cpu')
# print(z) # [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]