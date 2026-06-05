# ternary-mutual-info

**Mutual information: how much do two ternary signals know about each other?**

Two signals are correlated if they tend to move together. But correlation only captures linear relationships. Mutual information captures *any* relationship — linear, nonlinear, periodic, whatever. If knowing signal X tells you anything at all about signal Y, the mutual information is positive.

For ternary signals in `{-1, 0, +1}`, mutual information has a clean formulation: compute the joint distribution (a 3×3 table), the marginals, and then MI = Σ P(x,y) log₂(P(x,y) / (P(x)P(y))). The maximum MI is log₂(3) ≈ 1.585 bits — achieved when Y is a deterministic function of X.

This crate computes MI, normalized MI (0-1 scale), entropy, joint distributions, and total correlation across multiple sequences.

## What's Inside

- **`joint_distribution(x, y)`** — the 3×3 probability table for two ternary sequences
- **`marginals(table)`** — marginal distributions P(X) and P(Y)
- **`entropy(probs)`** — Shannon entropy in bits
- **`mutual_information(x, y)`** — MI in bits. 0 = independent, 1.585 = perfectly coupled
- **`normalized_mi(x, y)`** — MI normalized to [0, 1]
- **`total_correlation(sequences)`** — average pairwise MI across N sequences. Measures global coupling
- **`conditional_mi(x, y, z)`** — MI(X;Y|Z): how much does X tell you about Y *given* Z

## Quick Example

```rust
use ternary_mutual_info::*;

// Two identical signals: MI should be maximum
let x = vec![1, -1, 0, 1, -1, 0];
let y = vec![1, -1, 0, 1, -1, 0];
let mi = mutual_information(&x, &y);
assert!(mi > 1.5); // nearly log₂(3) — perfect coupling

// Two independent signals: MI should be near 0
let a = vec![1, 1, 1, -1, -1, -1];
let b = vec![-1, 0, 1, -1, 0, 1];
let mi2 = mutual_information(&a, &b);
assert!(mi2 < 0.5); // weak coupling

// Normalized: 0 to 1 scale
let nmi = normalized_mi(&x, &y);
assert!(nmi > 0.9);

// Total correlation: how coupled is a fleet of agents?
let fleet = vec![
    vec![1, -1, 0, 1],
    vec![1, -1, 0, -1],
    vec![1, 0, 0, 1],
];
let tc = total_correlation(&fleet);
println!("Fleet coupling: {:.3} bits", tc);
```

## The Deeper Truth

**For ternary signals, MI has exactly the right resolution.** The maximum MI is log₂(3) ≈ 1.585 bits — not 1 bit (binary) and not infinity (continuous). This means ternary MI distinguishes between three levels of coupling: zero (independent), moderate (some shared structure), and maximum (one is a function of the other). The 3×3 joint distribution has only 9 entries — small enough to compute exactly, large enough to capture non-trivial structure.

The total correlation (average pairwise MI) is the right metric for fleet health. If all agents are doing the same thing, TC is high (monoculture). If they're independent, TC is low (diverse). The sweet spot is in between — enough coupling for coordination, enough independence for resilience.

**Use cases:**
- **Feature selection** — which ternary features carry the most information about the target?
- **Multi-agent analysis** — how coupled is your agent fleet?
- **Causal inference** — high MI is a prerequisite for causation
- **Sensor fusion** — which sensors are redundant (high MI) vs. complementary (low MI)?
- **Information theory education** — the simplest non-binary MI computation

## See Also

- **ternary-entropy** — entropy computation (the building block of MI)
- **ternary-complexity** — Kolmogorov complexity (a different kind of information measure)
- **ternary-drift** — population coupling through genetic drift
- **ternary-consensus** — the dynamics that create high MI

## Install

```bash
cargo add ternary-mutual-info
```

## License

MIT
