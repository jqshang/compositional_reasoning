import numpy as np
import random


class CompositionalDataset:

    def __init__(self,
                 num_samples=1000,
                 min_seq_len=2,
                 max_seq_len=5,
                 input_dim=5,
                 train_ratio=0.8,
                 seed=42):

        np.random.seed(seed)
        random.seed(seed)

        self.num_samples = num_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.train_ratio = train_ratio

        self.basic_functions = {
            "inc": lambda x: x + 1,
            "dec": lambda x: x - 1,
            "mul2": lambda x: x * 2,
            "div2": lambda x: x / 2,
            "sort": lambda x: np.sort(x),
            "rev": lambda x: x[::-1],
        }
        self.function_names = list(self.basic_functions.keys())

    def generate_sample(self, composition):
        """Generate a single sample given a composition of functions."""
        x = np.random.randint(1, 100, self.input_dim).astype(float)
        intermediate_steps = [x.copy()]

        for func_name in composition:
            func = self.basic_functions[func_name]
            x = func(x)
            intermediate_steps.append(x.copy())

        return {
            "input": intermediate_steps[0],
            "output": intermediate_steps[-1],
            "composition": composition,
            "intermediate_steps": intermediate_steps[1:-1]
        }

    def generate_dataset(self, num_samples, seen_compositions=None):
        dataset = []

        for _ in range(num_samples):
            if seen_compositions:
                composition = random.choice(seen_compositions)
            else:
                seq_len = random.randint(self.min_seq_len, self.max_seq_len)
                composition = [
                    random.choice(self.function_names) for _ in range(seq_len)
                ]
            sample = self.generate_sample(composition)
            dataset.append(sample)

        return dataset

    def prepare_datasets(self):
        num_train_samples = int(self.num_samples * self.train_ratio)
        num_test_samples = self.num_samples - num_train_samples

        seen_compositions = []
        for _ in range(10):
            seq_len = random.randint(self.min_seq_len, self.max_seq_len)
            composition = [
                random.choice(self.function_names) for _ in range(seq_len)
            ]
            seen_compositions.append(composition)

        train_dataset = self.generate_dataset(
            num_train_samples, seen_compositions=seen_compositions)
        test_dataset = self.generate_dataset(num_test_samples)
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "seen_compositions": seen_compositions,
            "basic_functions": self.function_names
        }
