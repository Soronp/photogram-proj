import torch

print("=" * 60)
print("PYTORCH + CUDA CHECK")
print("=" * 60)

print("Torch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)

print("\nCUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("\n❌ CUDA is NOT available → GPU not usable")
    exit()

# GPU info
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)

print("\nGPU:", name)
print("Compute Capability:", cap)

# Arch support (THIS is the key check)
try:
    arch_list = torch.cuda.get_arch_list()
    print("\nSupported arch list:", arch_list)

    if "sm_120" in arch_list:
        print("\n✅ SUCCESS: sm_120 is supported!")
    else:
        print("\n❌ PROBLEM: sm_120 NOT in supported arch list")
except Exception as e:
    print("Could not read arch list:", e)

# Actual GPU usage test (important)
print("\nRunning small GPU test...")

try:
    x = torch.rand(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("✅ GPU computation SUCCESS")
except Exception as e:
    print("❌ GPU computation FAILED:", e)

print("=" * 60)