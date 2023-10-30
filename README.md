# Tinygrad Experiments

[Tinygrad](https://github.com/tinygrad/tinygrad) is cool... or so I've heard. Going to mess around with it here. Will update.

Goal is to get a basic understanding of how it works, and then try to contribute somehow to the project.

Update submodules:
    `git submodule update --remote`


## COMPLETED
- [X] follow the basic MLP example from the tinygrad README, get it training
- [X] train a model on MNIST or something
- [X] work through all the docs
  - [X] quickstart.md
  - [X] abstractions.py
- [X] annotate mlops.py
- [X] I think I should go and annotate the tensor class
  - [ ] line 470 -> end still TODO
- [X] Build simple dataloader

## TODO

### "Higher"
- [ ] replicate `tinygrad/examples/simple_conv_bn.py` from scratch with MNIST
- [ ] try to implement a transformer from scratch
  - [ ] OPTIONAL - reverse engineer and annotate the transformer example

### "Lower"
- [ ] try some different backends, compare
- [ ] reverse engineer the symbolic shape library
- [ ] reverse engineer the AST linearizer (codegen)

- [ ] check out some of the [issues on GitHub](https://github.com/tinygrad/tinygrad/issues)
  - See the [CONTRIBUTING.md](https://github.com/tinygrad/tinygrad/blob/c7f4dd6cb0651ad974f88a4ff2cf7dfe71c5d769/CONTRIBUTING.md)

## Hopes and Dreams
- [ ] try something cool like sliding window attention, flash attention, rotary (RoPE) embeddings, speculative decoding etc.
- [ ] speed up something with this [linalg paper](https://arxiv.org/abs/2309.03060)
  - [ ] reverse engineer some of [these operator abstractions](https://github.com/wilson-labs/cola) to see how they work 