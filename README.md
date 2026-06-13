# ternary-mutual-info

Mutual information estimation for ternary {-1, 0, +1} sequences. Pairwise MI, normalized MI, total correlation, MI matrices, and Shannon entropy — all in pure Rust with zero dependencies.

## Why It Matters

Mutual information measures the statistical dependence between two random variables. For ternary agent systems, MI quantifies how much knowing one agent's state reduces uncertainty about another — critical for:
- **Detecting coordination** between agents without observing their communication channel
- **Feature selection**: identifying which ternary signals carry redundant vs. complementary information
- **Information-theoretic causality**: MI is the foundation of transfer entropy and Granger causality on discrete state spaces
- **Compression bounds**: the total correlation sets the theoretical limit for lossless compression of multi-agent state

## How It Works

### Joint Distribution

For two ternary sequences $X, Y \in \{-1, 0, +1\}^n$, compute the 3×3 joint probability table:

$$P(x, y) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[X_i = x \wedge Y_i = y]$$

### Shannon Entropy

$$H(X) = -\sum_{x \in \{-1,0,+1\}} P(x) \log_2 P(x)$$

For a uniform ternary distribution, $H_{\max} = \log_2(3) \approx 1.585$ bits.

### Mutual Information

$$I(X; Y) = \sum_{x,y} P(x,y) \log_2 \frac{P(x,y)}{P(x) \cdot P(y)}$$

**Properties:**
- $I(X;Y) \geq 0$ (non-negative)
- $I(X;X) = H(X)$ (self-information equals entropy)
- $I(X;Y) = 0$ iff $X \perp Y$

**Complexity:** O(n) for joint distribution, O(1) for MI from the 3×3 table.

### Normalized MI

$$\text{NMI}(X,Y) = \frac{I(X;Y)}{(H(X) + H(Y)) / 2}$$

Ranges from 0 (independent) to 1 (identical distributions).

### Total Correlation

Multi-variable generalization — the average pairwise MI:

$$\text{TC} = \frac{2}{k(k-1)} \sum_{i < j} I(X_i; X_j)$$

**Complexity:** O(k² · n) for $k$ sequences of length $n$.

### MI Matrix

Symmetric $k \times k$ matrix with $M_{ij} = I(X_i; X_j)$ and diagonal $M_{ii} = H(X_i)$.

## Quick Start

```rust
use ternary_mutual_info::*;

let x = vec![1i8, 0, -1, 1, 0, -1, 1, 0, -1];
let y = vec![1i8, 0, -1, 1, 0, -1, 1, 0, -1];

// Mutual information (bits)
let mi = mutual_information(&x, &y);
assert!(mi > 1.0); // identical sequences have high MI

// Normalized MI (0-1)
let nmi = normalized_mi(&x, &y);
assert!((nmi - 1.0).abs() < 0.01); // identical → NMI = 1.0

// Total correlation across multiple sequences
let seqs = vec![x.clone(), y, vec![-1, 0, 1, -1, 0, 1, -1, 0, 1]];
let tc = total_correlation(&seqs);

// MI matrix
let m = mi_matrix(&seqs);
assert_eq!(m.len(), 3);
```

## API

| Function | Description |
|---|---|
| `joint_distribution(x, y) → [[f64;3];3]` | 3×3 joint probability table |
| `marginals(table) → ([f64;3], [f64;3])` | Marginal distributions from joint |
| `entropy(probs) → f64` | Shannon entropy in bits |
| `mutual_information(x, y) → f64` | MI between two ternary sequences |
| `normalized_mi(x, y) → f64` | Normalized MI ∈ [0, 1] |
| `total_correlation(seqs) → f64` | Average pairwise MI across sequences |
| `mi_matrix(seqs) → Vec<Vec<f64>>` | Full symmetric MI matrix |

## Architecture Notes

MI connects to the **γ + η = C** conservation identity through the information budget of ternary systems. For a ternary agent, the entropy $H(X) \leq \log_2 3$ decomposes into three components: the entropy of the positive mass distribution (γ), the entropy of the negative mass distribution (η), and the entropy of the neutral carrier (0). The mutual information $I(X; Y)$ measures how much of the conserved quantity $C$ is *shared* between agents — when $I$ is high, the agents' γ and η components are synchronized; when $I \approx 0$, they carry independent information about the conserved quantity.

The total correlation provides an upper bound on compression: if $k$ ternary agents have total correlation TC, then the joint state can be compressed from $k \cdot \log_2 3$ bits down to approximately $k \cdot \log_2 3 - \text{TC}$ bits.

## References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory.* 2nd ed. Wiley.
- Witten, I. H. & Frank, E. (2005). *Data Mining.* Sec. 2.7 (Normalized MI).
- Studený, M. & Vejnarová, J. (1998). *The Multiinformation Function as a Tool for Measuring Stochastic Dependence.* MIT Press.

## License

MIT
