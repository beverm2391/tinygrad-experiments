from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Conv2d, BatchNorm2d
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


class ConvBlock:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.conv_layer = Conv2d(input_channels, output_channels, kernel_size)
        self.batch_norm_layer = BatchNorm2d(output_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layer(x) # (batch_size, 28, 28, 1) -> (batch_size, 26, 26, 32)
        x = self.batch_norm_layer(x).relu()
        return x
    
    def __call__(self, x): return self.forward(x)

    def parameters(self) -> list:
        return get_parameters(self.conv_layer) + get_parameters(self.batch_norm_layer)


class TinyConv:
    def __init__(self):
        self.conv1 = ConvBlock(1, 32, 3)  # (batch_size, 1, 28, 28) -> (batch_size, 32, 26, 26)
        self.conv2 = ConvBlock(32, 64, 3)  # (batch_size, 32, 13, 13) -> (batch_size, 64, 11, 11)
        self.fc1 = Linear(64 * 5 * 5, 128)  # (batch_size, 1600) -> (batch_size, 128)
        self.fc2 = Linear(128, 10)  # (batch_size, 128) -> (batch_size, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 1, 28, 28)  # (batch_size, 784) -> (batch_size, 1, 28, 28)
        x = self.conv1(x)  # (batch_size, 1, 28, 28) -> (batch_size, 32, 26, 26)
        x = x.max_pool2d(kernel_size=(2,2))  # (batch_size, 32, 26, 26) -> (batch_size, 32, 13, 13)
        x = self.conv2(x)  # (batch_size, 32, 13, 13) -> (batch_size, 64, 11, 11)
        x = x.max_pool2d(kernel_size=(2,2))  # (batch_size, 64, 11, 11) -> (batch_size, 64, 5, 5)
        x = x.reshape(x.shape[0], -1)  # (batch_size, 64, 5, 5) -> (batch_size, 1600)
        x = self.fc1(x).relu()  # (batch_size, 1600) -> (batch_size, 128)
        x = self.fc2(x).softmax()  # (batch_size, 128) -> (batch_size, 10)

        return x
    
    def __call__(self, x): return self.forward(x)

    def parameters(self) -> list:
        return get_parameters(self.conv1) + get_parameters(self.conv2) + get_parameters(self.fc1) + get_parameters(self.fc2)