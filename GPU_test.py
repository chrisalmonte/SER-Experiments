import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("device", "cuda")
    print("GPU_count", torch.cuda.device_count())
    print("GPU_device", torch.cuda.get_device_name(0))
else:
    print("device", "cpu")