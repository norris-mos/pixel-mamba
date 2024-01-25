from datasets import interleave_datasets, load_dataset


dataset = load_dataset("wikipedia",'20220301.en', split="train", streaming=True)

sample = next(iter(dataset.take(5)))
print(len(sample))
print(sample)