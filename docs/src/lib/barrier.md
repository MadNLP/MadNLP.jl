```@meta
CurrentModule = MadNLP
```

# Barrier strategies

The structure `AbstractBarrierUpdate` encodes the barrier strategy currently
used in MadNLP. The strategy is defined by the option `barrier` when initializing
MadNLP.
```@docs
AbstractBarrierUpdate
update_barrier!

```

## Monotone strategy

By default, MadNLP uses a monotone strategy, following the Fiacco-McCormick approach.

```@docs
MonotoneUpdate

```

## Adaptive strategies

As an alternative to the monotone strategy, MadNLP implements several
adaptive rules, all described in [this article](https://epubs.siam.org/doi/abs/10.1137/060649513).
```@docs
AbstractAdaptiveUpdate
QualityFunctionUpdate
LOQOUpdate

```
