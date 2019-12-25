### 1. Tensor

I learn the basic concepts about tensor from *Pytorch*. A tensor maintains four things, storage address, offset, stride and shape, which entirely determine the tensor.

```c++
struct Tensor {
    Storage<Dtype> storage_;
    Shape<Dim> shape_;
    int stride_[Dim];
}
```

#### 1.1 Storage

`Storage<Dtype> storage_` holds the data of a tensor. The object `storage_` doesn't be shared between different tensors, but the actual storage space does.  I use `shared_ptr<Dtype> bptr_` to maintain storage address and `Dtype* dptr_ = bptr_ + offset` to maintain offset.

I think `shared_ptr` can make me free of annoying `delete`, but I'm not sure this is right, because I notice that in *Pytorch*, it is used that `unique_prt` instead of `shared_ptr`, but I don't really look deep into it.

#### 1.2 Shape

`Shape<Dim> shape_` can be viewed as `int shape_[Dim]`. It holds the size on each dimension.

#### 1.3 Stride

`int stride_[Dim]` hold the stride on each dimension, which is the most interesting thing about a tensor. For a D dimension tensor, storage address of the idx element can be calculated as follows:
$$
Addr_k = dptr + \sum^{k-1}_{i = 0}stride[i] * idx[i]
$$
where $dptr$ is the tensor' storage address. With the concept of stride, many operation can be implemented efficiently. Take 3D tensor as example and explain this with pseudo code.

- `b = a[:, i:j, :]` 

    ```c++
    b.dptr = a.dptr + i * a.stride[1];
    b.shape = a.shape.copy(); b[1] = j - i;
    b.stride = a.stride.copy()
    ```

- `b = a[:, i, :]`

    ```c++
    b.dptr = a.dptr + i * a.stride[1];
    b.shape = new int[2]; b.shape[0] = a.shape[0]; b.shape[1] = a.shape[2];
    b.stride = new int[2]; b.stride[0] = a.stride[0]; b.stride[1] = a.stride[2];
    ```

- `b = a.transpose(i, j)`

    ```c++
    b.dptr = a.dptr;
    b.shape = a.shape.swap(i, j);
    b.stride = a.stride.swap(i, j);
    ```

- `c = a + b  // broadcasting:　Shape(i, j, k) + Shape(i, 1, k)`

    ```c++
    // set a.stride as normal
    a.stride[0] = j*k; a.stride[1] = k; a.stride[2] = 1;
    // set b.stride[x] = 0 if b.shape[x] == 1
    b.stride[0] = j*k; b.stride[1] = 0; b.stride[2] = 1; 
    // when calculating the result, y is meaningless for b
    c[x, y, z] = a[x, y, z] + b[x, y, z]
    ```

### 2. Template Expression

At the beginning, I write codes of this part following the exactly same way as [mshadow's exp-template tutorial](https://github.com/dmlc/mshadow/tree/master/guide/exp-template). Then, I simplified these codes by inherit, which means use dynamic binding instead of static binding. Static binding is more efficient, while dynamic binding will make code more simple.

Matrix Multiply is different from element-wise operation, which should be implemented in a different way. But it's a pity that I implement MM in the same way as element-wise operation, which will cause unnecessary computation. I did so for make codes clear, and maybe fix it one day.

### 3. Hierarchy of Abstraction

Generally, there are two main levels, Tensor and Node.

In Tensor level, from bottom to top, there are **Expression**, **Tensor**, and **Operation function**. This level has no relationship to computation graph. So when only use Tensor, the operations can't backward.

In Node level, computation won't be processed, which is done in tensor level. Here dynamic computation graph is constructed when computation flow forwards. So all neural network modules are on base of Node.

