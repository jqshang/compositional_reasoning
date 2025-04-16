# Revised model code (scripts/function_composition.py)
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SortModule(nn.Module):
    def forward(self, x):
        return torch.sort(x, dim=-1)[0]

class ReverseModule(nn.Module):
    def forward(self, x):
        return torch.flip(x, dims=[-1])

class AddModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x + self.scalar

class SubtractModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x - self.scalar

class MultiplyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x * self.scalar

class DivideModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x / (self.scalar + 1e-5)

class CompositionController(nn.Module):
    def __init__(self, input_size, hidden_size, num_modules, sequence_length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.module_selector = nn.Linear(hidden_size, num_modules * sequence_length)
        self.sequence_length = sequence_length
        self.num_modules = num_modules
    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        logits = self.module_selector(encoded).view(batch_size, self.sequence_length, self.num_modules)
        return logits

class HierarchicalCompositionalModel(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super().__init__()
        self.modules_list = nn.ModuleList([
            SortModule(),
            ReverseModule(),
            AddModule(),
            SubtractModule(),
            MultiplyModule(),
            DivideModule()
        ])
        self.sequence_length = sequence_length
        self.controller = CompositionController(input_size, hidden_size, len(self.modules_list), sequence_length)
    def forward(self, x):
        module_logits = self.controller(x)
        module_probs = F.softmax(module_logits, dim=-1)
        current_output = x
        outputs = []
        for step in range(self.sequence_length):
            module_idx = torch.argmax(module_probs[:, step, :], dim=-1)
            next_output = []
            for i, idx in enumerate(module_idx):
                next_output.append(self.modules_list[idx](current_output[i]))
            next_output = torch.stack(next_output, dim=0)
            outputs.append(next_output)
            current_output = next_output
        outputs = torch.stack(outputs, dim=1)
        return outputs, module_logits
