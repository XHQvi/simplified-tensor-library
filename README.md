# simplified-tensor-library
This repository is mainly a C++ practice. Some basic concepts about tensor, template expression and neural network are implemented. I try to make the codes simple and easy to understand, instead of making it efficient.

### Features

1. Tensor's basic concepts are implemented, such as storage, stride and size. And operations like transpose, slice and view can be processed with negligible computational cost.
2. Template expression are implemented, so expression can be computed by a lazy way and avoid extra temporary storage allocation.
3. Dynamic computation graph mechanism is implemented. So forward and backward for most operations are available.
4. Basic neural network training tools, SGD optimizer, Kaiming uniform init, CrossEntropyLoss  and so on, are implemented.

### build and run

The codes are totally simple, so I didn't use `make` tools . Just complie the files using `g++` or something else. The script `./eleven/scripts/build_lenet.sh` will do the same thing.

I implemented a MLP with three FC layers and a smaller LeNet and trained them on MNIST dataset for only one epoch . As I said above, these codes are not efficient. It will take a quite long time to train them, especially MLP. The result is as follows.

|      model       |  acc   |
| :--------------: | :----: |
|  Smaller LeNet   | 0.9183 |
| Three layers MLP | 0.9415 |

### about codes

More details about implement can be found in `./eleven/guide.md`.

