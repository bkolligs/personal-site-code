# Spatial Transforms

The code used in the [Spatial Transforms](https://benkolligs.com/notes/Robotics/Spatial-Transforms) note.

## Prerequisites

In order to run this code, you will need the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager. 
You can quickly install it with: 

```
curl -LsSf https://astral.sh/uv/install.sh | sh

```

## Usage

The [main script](./main.py) has three sub parsers: 
1. `frames`: runs the multi-frame transform
2. `inverse`: runs the inverse test 
3. `einsum`: runs the einsum example

We can run the script with `uv`:
```bash
uv run main.py -h
```
