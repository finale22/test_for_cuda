import torch

def test_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

if __name__ == "__main__":
    test_cuda()
