# vfann: V Bindings for Fast Artificial Neural Networks(FANN) library

> [FANN](https://github.com/libfann/fann) is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks. This project provides V bindings for the FANN library.
>

## Documentation

Documentation for this project can be found over [here](/docs/vfann.md).

## Installation
- Install `libfann`
You can install libfann binary package via package manager:
```sh
 sudo apt install libfann-dev
```
Or you can install it via github
```sh
 git clone https://github.com/libfann/fann
```

- Install `vfann`
```sh
 v install --git https://github.com/kbkpbot/vfann
```
## Example
```v
module main

import vfann as fann

fn main() {
        ann := fann.create_from_file('xor_float.net')

        input := [f32(-1), 1]
        calc_out := ann.run(input.data)

        println('xor test (${input[0]},${input[1]}) -> ${calc_out[0]}')

        ann.destroy()
}
```

Please check the [examples](/examples) directory for more examples.

## License

MIT license
