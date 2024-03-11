import torch
import quiptools


def benchmark():
    torch.manual_seed(42)
    N = 12288
    K = 4096

    x = torch.randn((K,), dtype=torch.float16, device="cuda")
    Qidxs = torch.randint(1 << 15, (N // 16, K // 64, 8, 4),
                          dtype=torch.int64,
                          device="cuda")
    codebook = torch.randint(0x7FFFFFFF, (256,),
                             dtype=torch.int32,
                             device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    x = quiptools.decode_matvec_e8p(x, Qidxs-0x8000, codebook)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed: {elapsed_time_ms:.4f}ms")


if __name__ == "__main__":
    benchmark()
