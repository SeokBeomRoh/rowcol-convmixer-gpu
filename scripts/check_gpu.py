import torch


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("gpu count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
    else:
        print("CUDA is not available. Check NVIDIA driver, WSL GPU support, and PyTorch CUDA wheel installation.")


if __name__ == "__main__":
    main()
