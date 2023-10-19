import torch, argparse, time
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./bert-large-conll2003/training")
parser.add_argument("--sequence-length", type=int, default=384)
parser.add_argument("--dtype", type=str, default="fp16")

BATCH_SIZES = [1,64,256]

def main(model_path, sequence_length, dtype):
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError("--dtype must be fp16, bf16, or fp32")

    print("------- LOADING MODEL --------")
    bert = AutoModelForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch_dtype
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("------- LOADING DATA --------")
    dataset = load_dataset("imdb", split="test")

    IDX = 3

    batches = [
        tokenizer(
            dataset[IDX:IDX+batch_size]["text"], 
            return_tensors="pt", 
            max_length=sequence_length, 
            padding="max_length", truncation=True
        )
    for batch_size in BATCH_SIZES]

    for batch in batches:
        for k in batch:
            batch[k] = batch[k].cuda()
    
    print("------- WARMING UP -------")
    bert.eval()
    for _ in range(10):
        _ = bert(**batches[0])
    torch.cuda.synchronize()

    bert.eval()
    with torch.no_grad():
        for batch in batches:
            batch_size = batch["input_ids"].shape[0]
            iterations = 1000 if batch_size == 1 else 10
            
            print(f"------- STARTING B={batch_size} -------")
            print(batch["input_ids"].shape)

            start = time.perf_counter()
            for _ in range(iterations):
                output = bert(**batch)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_items = iterations * batch_size
            total_time = end - start
            print(f"TOTAL_ITEMS = {total_items}")
            print(f"TOTAL_TIME = {total_time :0.2f}")
            print(f"THROUGHPUT = {total_items / total_time :0.2f}")

if __name__ == "__main__":
    args = parser.parse_args()

    main(
        args.model_path, 
        args.sequence_length,
        args.dtype
    )