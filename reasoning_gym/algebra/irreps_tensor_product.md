# O(3) Irreps Tensor Product Task

## Task Summary

`irreps_tensor_product` is a small algebra dataset for binary O(3) irrep tensor-product decomposition using e3nn-style single-irrep notation such as `2e` and `3o`.

## Mathematical Rule

For two single irreps `(l1, p1)` and `(l2, p2)`, the output contains every `l` from `|l1 - l2|` through `l1 + l2`, inclusive, with output parity `p1 * p2`.

## Parity Multiplication

- `e × e = e`
- `e × o = o`
- `o × e = o`
- `o × o = e`

## Canonical Answer Format

Answers use ascending canonical terms separated by ` + `.

Examples:

- `2e x 3o -> 1o + 2o + 3o + 4o + 5o`
- `1o x 1o -> 0e + 1e + 2e`
- `0e x 4o -> 4o`

## Scoring Contract

Accepted:

- `1o + 2o + 3o`
- `1o+2o+3o`
- `1o ⊕ 2o ⊕ 3o`

Rejected:

- `1o, 2o, 3o`
- `(1o, 2o, 3o)`
- `[1o, 2o, 3o]`
- `1o 2o 3o`
- `answer: 1o + 2o + 3o`

The scorer normalizes only superficial presentation differences:

- leading and trailing whitespace
- arbitrary whitespace around separators
- `⊕` as an alternate decomposition separator

It does not normalize structural syntax such as commas, tuple/list notation, bare whitespace-separated terms, or prose wrappers.

## Config Knobs

- `min_l`
- `max_l`
- `include_even`
- `include_odd`
- `seed`
- `size`

## First PR scope

This first PR supports only binary tensor products of two single irreps. It does not support direct sums, multi-factor products, or inputs where either side is itself a sum of irreps.

## First PR Non-goals

- curriculum
- export stratification
- multi-irrep input expressions
- natural-language answer parsing

## Dependency Note

This task uses e3nn-style O(3) irrep notation as reference context only. The v1 generator does not depend on `e3nn` at runtime because the accepted task contract only requires parsing single irrep tokens, multiplying parities, and constructing the inclusive `l`-range decomposition.

Adding `e3nn` would unnecessarily expand the dependency surface for a small exact symbolic dataset while not changing the oracle behavior of this first PR.

## Minimal Generated Example

Example prompt shape:

> Decompose the O(3) tensor product `2e x 3o`. Use this rule: output every irrep from `|l1-l2|` through `l1+l2`, inclusive, and multiply parity using `e*e=e, e*o=o, o*e=o, o*o=e`.

Example answer:

> `1o + 2o + 3o + 4o + 5o`
