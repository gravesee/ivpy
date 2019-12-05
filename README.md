
<img align="left" width="60" height="60" src="ivy.png">

# ivpy

A package with a single function. It's entire purpose is to return a set of break points that can be used to discretize a continuous variable. It does this through recursive partitioning using information value.

## Features

- Discretize continuous arrays using a supervised algorithm guided by information value.
- Many options to guide discretization process:
  - min information value for a split
  - min observation weights in bin
  - min response value weights in a bin
  - maximum number of bins
  - increasing and decreasing monotonicity
  - exception values
- speed

## Limitations

- Only binary response variables supported

## Examples

Examples of usage can be found in the [examples](examples/example02.ipynb) folder.
