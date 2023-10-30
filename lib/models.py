from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters


class TinyMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Linear(input_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, output_dim)

    def forward(self, x) -> Tensor:
        return self.l2(self.l1(x).relu())
    
    def parameters(self):
        return get_parameters(self.l1) + get_parameters(self.l2)
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Tiny MLP({self.l1}, {self.l2})"