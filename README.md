# Batch Processing

A Python library to make batchable Python scripts.
Main features are:

- Collects and filters any number of input files from the disk.
- Parallelisation using multiprocessing pool.
- OS-independent paths (thanks to the [Unified Path](https://github.com/emmcb/unified-path) library).
- Saves at each run the parameters of the run in a timestamped JSON file.
- Can use serialized input parameters in addition of the command line arguments.
- Can replicate arbitrary input folder structures in the output folder.
- Can handle multiple input files for one output.

## Installation

Install from pypi using pip:

```
pip install batch-processing
```

## Batch scripts

Here is a minimal example of a Batch script:

```python
import argparse
import sys
from pathlib import Path

from batch_processing import Batch


def parse_command_line(batch: Batch):
    parser = argparse.ArgumentParser(description='Batch example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Setup custom script arguments here

    return batch.parse_args(parser)


def parallel_process(input: Path, output: Path, args: argparse.Namespace, additional_argument: float):

    # Load an input file, apply some processing, and save to output file here

    return result  # optionally return some result


def main(argv):
    # Instantiate batch
    batch = Batch(argv)
    batch.set_io_description(input_help='input files', output_help='output directory')

    # Parse arguments
    args = parse_command_line(batch)

    # Optionally do some pre-processing here before
    additional_argument = 5.0

    # Start processing and retrieve the results at the end if any
    data = batch.run(parallel_process, additional_argument, output_ext=Batch.USE_INPUT_EXT)

    # Optionally do some post processing here


if __name__ == "__main__":
    main(sys.argv[1:])
```

## Usage

Arguments are defined using the standard `argparse` Python library.
As soon as the Batch library is used, predefined arguments are automatically added to the argparse parser.

The complete list of accepted arguments can be obtained by running a script using the Batch library with the `-h` or `--help` flag.

### Inputs

Any number of inputs can be given using the `-i` flag. Simple wildcard `*` and recursive wildcard `**` are accepted.

```
-i input1 input2 input3
-i "folder1/**/*.png" "folder2/**/*.png"
```

#### Tags

The inputs can additionaly be categorized using tags. Tags are defined by appending a word that begins with + in the input list.

```
-i +tag1 "folder1/**/*.png" +tag2 "folder2/**/*.png"
```

#### Filtering

The `--only` flag allows to select the inputs that match the given patterns, whereas `--exclude` allows to reject them.

```
-i **/*.png --exclude "regex1" "regex2" --only "regex1" "regex2"
```

#### Grouping

Inputs can be grouped using the `--group-inputs` argument.
The inputs that should be grouped together must share a common key, that is computed from the regex provided in the argument.

### Ouputs

Output can be given using the `-o` flag. Depending on the batch operation mode, the output can either be a folder or a file.

The additional `--prefix` and `--suffix` flags allow a append a prefix and/or a suffix to the output filename.

### Root directory

Inputs can be given relative to a root directory. That is to say, it is possible to call:

```
--root-dir D:/Images -i "folder1/*.jpg" "folder2/*.jpg"
```

Instead of:

```
-i "D:/Images/folder1/*.jpg" "D:/Images/folder2/*.jpg"
```

The main usage is that it allows to use the special directive `:dir` in the arguments.
`:dir` represents the folder hierarchy between the root path and the considered file.

For example:

- `-o output_dir/:dir` will replicate the input folder hierarchy in the output folder (we will get outputs for `folder1` inputs in `output_dir/folder1`, and outputs for `folder2` inputs in `output_dir/folder2`).
- `-o output_dir --prefix :dir` will flatten the hierarchy in output_dir, but will prefix the outputs by the folder (for example output corresponding to `folder1/img1.jpg` will be `output_dir/folder1_img1.jpg`).

### Multiprocessing

When multiprocessing is used, the number of parallel processes can be chosen with the `-j` flag.
By default, it is fixed to the number of CPU cores + 2, capped to 12.

### Logging

The batch library automatically initializes the `logging` standard Python library.
The `--log-level` flag allows to display messages up to the given level (by default `info`), while the `--log-file` flag allows
to redirect the log output to a file.

```
--log-level debug --log-file logs.txt
```

### Serialization

All arguments (batch arguments and custom script arguments) can be serialized in a JSON file.
The JSON keys correspond to the `dest` values of the argparse argument definitions.

```json
{
    "args": {
        "alpha": 0.5,
        "beta": 0.7
    }
}
```

Serialized argument files can be given on the command line with the `-a` argument.

```
-i inputs -o output -a args.json
```

#### Priority rules

If multiple JSON files are provided, they will be merged together along with the command line arguments.
When an argument appears multiple times, the priority (higher to lower) is:

1. The command line.
2. The serialized files from right to left.

```
-a first.json second.json --beta 0.9
```

In this example, the `beta` value from `second.json` will override the `beta` value from `first.json`.
Finally, the `--beta 0.9` argument will override the `beta` values from both `first.json` and `second.json`.

#### Recursion

A JSON argument file can itself refer to another JSON.

```json
{
    "args": {
        "alpha": 0.5,
        "arg_files": ["data.json"]
    }
}
```

In this case, `"alpha": 0.5` will override the `alpha` value from `data.json`.

### Archiving

Each time a Batch script is ran, an unique JSON file is created in the output folder, tracking the current values of the arguments.
This is especially useful for:

- Debugging: we can see the actual parameters as parsed by the Batch library.
- Reproducing the results: we can use back this file as input (with `-a`) in order to launch again the script with the same arguments as archived.
- Archiving: it documents how the outputs have been generated, for future reference.
