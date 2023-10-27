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

## TODO
- [ ] annotate mlops.py
- [ ] I think I should go and annotate the tensor class
  - [ ] Maybe make some detailed docs or notes that could be useful to someone else learning the library

## Hopes and Dreams
- [ ] reverse engineer and annotate the transformer example
- [ ] try to implement a transformer from scratch
  - [ ] add something cool like sliding window attention, flash attention, rotary (RoPE) embeddings, speculative decoding etc.

## Notes

1. What's the point of LazyOps???
2. In `tensor.py`, `Function` is defined, which is imported by `mlops.py` The problem is, '`tensor.py` also imports `mlops.py`. This is seemingly solved by defining all objects imported by `mlops.py` before its import in `tensory.py`, avoiding a circular import.