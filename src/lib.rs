#![forbid(unsafe_code)]
//! Mutual information for ternary {-1,0,+1} sequences.

use std::collections::HashMap;

/// Compute joint probability table for two ternary sequences.
pub fn joint_distribution(x: &[i8], y: &[i8]) -> [[f64; 3]; 3] {
    let n = x.len().min(y.len()) as f64;
    let mut table = [[0.0f64; 3]; 3];
    for i in 0..x.len().min(y.len()) {
        let xi = (x[i] + 1) as usize;
        let yi = (y[i] + 1) as usize;
        if xi < 3 && yi < 3 { table[xi][yi] += 1.0; }
    }
    for i in 0..3 { for j in 0..3 { table[i][j] /= n; } }
    table
}

/// Compute marginal distributions.
pub fn marginals(table: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
    let mut px = [0.0; 3]; let mut py = [0.0; 3];
    for i in 0..3 { for j in 0..3 { px[i] += table[i][j]; py[j] += table[i][j]; } }
    (px, py)
}

/// Compute Shannon entropy from probabilities.
pub fn entropy(probs: &[f64]) -> f64 {
    probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum()
}

/// Compute mutual information between two ternary sequences.
pub fn mutual_information(x: &[i8], y: &[i8]) -> f64 {
    if x.is_empty() || y.is_empty() { return 0.0; }
    let table = joint_distribution(x, y);
    let (px, py) = marginals(&table);
    let mut mi = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            if table[i][j] > 0.0 && px[i] > 0.0 && py[j] > 0.0 {
                mi += table[i][j] * (table[i][j] / (px[i] * py[j])).log2();
            }
        }
    }
    mi.max(0.0)
}

/// Normalized mutual information (0 to 1).
pub fn normalized_mi(x: &[i8], y: &[i8]) -> f64 {
    let mi = mutual_information(x, y);
    let hx = entropy(&marginals(&joint_distribution(x, y)).0);
    let hy = entropy(&marginals(&joint_distribution(x, y)).1);
    let h = (hx + hy) / 2.0;
    if h == 0.0 { 0.0 } else { mi / h }
}

/// Total correlation across multiple sequences (average pairwise MI).
pub fn total_correlation(sequences: &[Vec<i8>]) -> f64 {
    if sequences.len() < 2 { return 0.0; }
    let mut sum = 0.0; let mut count = 0;
    for i in 0..sequences.len() {
        for j in (i+1)..sequences.len() {
            sum += mutual_information(&sequences[i], &sequences[j]);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// MI matrix for all pairs.
pub fn mi_matrix(sequences: &[Vec<i8>]) -> Vec<Vec<f64>> {
    let n = sequences.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n { matrix[i][i] = entropy(&ternary_probs(&sequences[i])); }
    for i in 0..n { for j in (i+1)..n { let mi = mutual_information(&sequences[i], &sequences[j]); matrix[i][j] = mi; matrix[j][i] = mi; } }
    matrix
}

fn ternary_probs(seq: &[i8]) -> Vec<f64> {
    let mut counts = [0usize; 3];
    for &v in seq { let idx = (v + 1) as usize; if idx < 3 { counts[idx] += 1; } }
    let n = seq.len() as f64;
    counts.iter().map(|&c| c as f64 / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_identical_sequences() { let s = vec![1,0,-1,1,0,-1]; assert!(mutual_information(&s, &s) > 1.0, "MI of identical should be high"); }
    #[test] fn test_independent_sequences() { let x = vec![1,-1,1,-1,1,-1]; let y = vec![-1,1,-1,1,-1,1]; let mi = mutual_information(&x, &y); assert!(mi < 0.1, "MI of anti-correlated should be low: {}", mi); }
    #[test] fn test_mi_symmetric() { let x = vec![1,0,-1]; let y = vec![-1,0,1]; assert!((mutual_information(&x, &y) - mutual_information(&y, &x)).abs() < 0.001); }
    #[test] fn test_entropy_uniform() { let probs = vec![1.0/3.0, 1.0/3.0, 1.0/3.0]; assert!((entropy(&probs) - 1.585 * 1.5).abs() < 0.01); }
    #[test] fn test_entropy_pure() { assert!((entropy(&[1.0, 0.0, 0.0])).abs() < 0.001); }
    #[test] fn test_joint_distribution_sums_to_one() { let x = vec![1,0,-1]; let y = vec![-1,0,1]; let t = joint_distribution(&x, &y); let sum: f64 = t.iter().flat_map(|r| r.iter()).sum(); assert!((sum - 1.0).abs() < 0.01); }
    #[test] fn test_normalized_mi_range() { let x = vec![1,0,-1,1,0,-1]; let y = vec![1,0,-1,1,0,-1]; let nmi = normalized_mi(&x, &y); assert!(nmi >= 0.0 && nmi <= 1.01); }
    #[test] fn test_total_correlation_single() { let seqs = vec![vec![1,0,-1]]; assert_eq!(total_correlation(&seqs), 0.0); }
    #[test] fn test_total_correlation_pair() { let seqs = vec![vec![1,0,-1], vec![1,0,-1]]; assert!(total_correlation(&seqs) > 0.5); }
    #[test] fn test_mi_matrix_square() { let seqs = vec![vec![1,0,-1], vec![-1,0,1], vec![1,1,1]]; let m = mi_matrix(&seqs); assert_eq!(m.len(), 3); assert_eq!(m[0].len(), 3); }
    #[test] fn test_mi_matrix_symmetric() { let seqs = vec![vec![1,0,-1], vec![-1,0,1]]; let m = mi_matrix(&seqs); assert!((m[0][1] - m[1][0]).abs() < 0.001); }
    #[test] fn test_empty_input() { assert_eq!(mutual_information(&[], &[]), 0.0); }
    #[test] fn test_constant_sequences() { let x = vec![1,1,1,1]; let y = vec![1,1,1,1]; let mi = mutual_information(&x, &y); assert!(mi < 0.01, "Constant sequences should have ~0 MI: {}", mi); }
    #[test] fn test_long_correlated() { let x: Vec<i8> = (0..100).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(); let y: Vec<i8> = x.iter().map(|&v| -v).collect(); let mi = mutual_information(&x, &y); assert!(mi > 0.5, "Perfectly anti-correlated should have MI: {}", mi); }
    #[test] fn test_marginals_sum() { let x = vec![1,0,-1,1]; let y = vec![0,1,-1,0]; let t = joint_distribution(&x, &y); let (px, py) = marginals(&t); assert!((px.iter().sum::<f64>() - 1.0).abs() < 0.01); assert!((py.iter().sum::<f64>() - 1.0).abs() < 0.01); }
    #[test] fn test_total_correlation_independent() { let seqs = vec![vec![1,-1,1,-1], vec![-1,1,-1,1], vec![1,1,-1,-1]]; let tc = total_correlation(&seqs); assert!(tc < 0.5, "Independent seqs should have low TC: {}", tc); }
    #[test] fn test_normalized_mi_identical_is_one() { let s = vec![1,0,-1,1,0,-1,1,0,-1]; assert!((normalized_mi(&s, &s) - 1.0).abs() < 0.01); }
    #[test] fn test_mi_nonnegative() { let x = vec![1,0,-1]; let y = vec![1,1,1]; assert!(mutual_information(&x, &y) >= 0.0); }
    #[test] fn test_ternary_probs() { let seq = vec![1,0,-1]; let p = ternary_probs(&seq); assert!((p[0] - 1.0/3.0).abs() < 0.01); }
}
