import numpy as np
import math
from time import perf_counter

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD

from lib.utils import get_mnist
from lib.models import TinyMLP


def main():

    X_train, Y_train, X_test, Y_test = get_mnist("../data")

    model = TinyMLP(784, 100, 10) # instantiate the model
    optim = SGD(model.parameters(), lr=0.001) # instantiate the optimizer

    EPOCHS = 10
    STEPS = 1000 # num of batches per epoch
    BATCH_SIZE = 64
    
    _max_batches_per_epoch = math.ceil(len(X_train) / BATCH_SIZE) # handle smaller last batch
    _steps = min(STEPS, _max_batches_per_epoch)

    # ! Train ================================================================

    total_time = 0.0

    for epoch in range(EPOCHS):
        start = perf_counter()
        running_train_loss = 0.0
        for step in range(_steps):
            with Tensor.train():
                samp = np.random.randint(0, X_train.shape[0], size=(64))

                # get batch and labels
                batch = Tensor(X_train[samp], requires_grad=False)
                labels = Tensor(Y_train[samp])

                out = model(batch) # forward pass
                loss = out.sparse_categorical_crossentropy(labels) # calculate loss
                optim.zero_grad() # zero out gradients
                loss.backward() # backward pass
                optim.step() # update weights

                running_train_loss += loss.numpy()

        train_loss = running_train_loss / STEPS # loss over all batches, over num batches

        # test accuracy over the whole dataset
        out = model(Tensor(X_test))
        pred = out.argmax(axis=1) # get the index of the max value
        accuracy = (pred == Tensor(Y_test)).mean().numpy()

        elapsed = perf_counter() - start
        total_time += elapsed

        print(f"Epoch {epoch+1}/{EPOCHS}: {_steps} Batches (max: {_max_batches_per_epoch}) | Train Loss: {train_loss:.4f} | Test Accuracy: {accuracy:.4f} | Time: {elapsed:.2f}s")

    print(f"Total training time: {total_time:.2f}s")

if __name__ == "__main__":
    main()