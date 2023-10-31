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
- [X] build a conv on MNIST
  - [ ] see how well you can get conv performing
- [ ] [here's some papers](https://paperswithcode.com/sota/image-classification-on-mnist) of best performing MNIST models
- [ ] try to implement a transformer from scratch
  - [ ] https://github.com/fkodom/transformer-from-scratch
  - [ ] https://fkodom.substack.com/p/transformers-from-scratch-in-pytorch
  - [ ] OPTIONAL - reverse engineer and annotate the transformer example
- [ ] find a new example model architecture to add to repo with PR
  - [ ] port something of [Lucidrains](https://github.com/lucidrains?tab=repositories) over to tinygrad
    - [ ] [vision transformers](https://github.com/lucidrains/vit-pytorch)
    - [ ] [x-transformers](https://github.com/lucidrains/x-transformers)
    - [ ] [x-clip](https://github.com/lucidrains/x-clip)
  - [ ] [fft convolution](https://github.com/fkodom/fft-conv-pytorch)

### "Lower"
- [ ] try some different backends, compare
- [ ] reverse engineer the symbolic shape library
- [ ] reverse engineer the AST linearizer (codegen)

- [ ] check out some of the [issues on GitHub](https://github.com/tinygrad/tinygrad/issues)
  - See the [CONTRIBUTING.md](https://github.com/tinygrad/tinygrad/blob/c7f4dd6cb0651ad974f88a4ff2cf7dfe71c5d769/CONTRIBUTING.md)

## Hopes and Dreams
- [ ] try something cool like sliding window attention, [flash attention](https://github.com/Dao-AILab/flash-attention), [rotary (RoPE) embeddings](https://github.com/lucidrains/rotary-embedding-torch), speculative decoding etc.
- [ ] speed up something with this [linalg paper](https://arxiv.org/abs/2309.03060)
  - [ ] reverse engineer some of [these operator abstractions](https://github.com/wilson-labs/cola) to see how they work 