use std::cmp::Ordering;
use super::Distance;

pub trait SparseVec<K> {
    fn idxs(&self) -> &[usize];
    fn vals(&self) -> &[K];
}

#[derive(Clone)]
pub struct Euclidean;

impl Distance<Vec<f32>> for Euclidean {
    fn distance(&self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        x.iter().zip(y.iter()).map(|xyi| (xyi.0 - xyi.1).powi(2)).sum()
    }
}

impl <R: SparseVec<f32>> Distance<R> for Euclidean {
    fn distance(&self, x: &R, y: &R) -> f32 {
        let mut i = 0usize;
        let mut j = 0usize;
        let mut sum = 0f32;
        let x_idxs = x.idxs();
        let x_vals = x.vals();
        let y_idxs = y.idxs();
        let y_vals = y.vals();
        while i < x_idxs.len() && j < y_idxs.len() {
            match x_idxs[i].cmp(&y_idxs[j]) {
                Ordering::Less => {
                    sum += x_vals[i].powi(2);
                    i += 1;
                },
                Ordering::Greater => {
                    sum += y_vals[j].powi(2);
                    j += 1
                },
                _ => {
                    sum += (x_vals[i] - y_vals[j]).powi(2);
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }
}

#[derive(Clone)]
pub struct Cosine(pub bool);

impl Distance<Vec<f32>> for Cosine {
    fn distance(&self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        let mut dot = 0f32;
        let mut x_norm = 0f32;
        let mut y_norm = 0f32;
        for (xi, yi) in x.iter().zip(y.iter()) {
            dot += xi * yi;
            if self.0 {
                x_norm += xi.powi(2);
                y_norm += yi.powi(2);
            }
        }

        if self.0 {
            let xy_norm = x_norm.sqrt() * y_norm.sqrt();
            dot /= if xy_norm > 0. { xy_norm } else { 1f32 };
        } 
        1f32 - dot
    }
}

impl <R: SparseVec<f32>> Distance<R> for Cosine {
    fn distance(&self, x: &R, y: &R) -> f32 {
        let mut i = 0usize;
        let mut j = 0usize;
        let x_idxs = x.idxs();
        let x_vals = x.vals();
        let y_idxs = y.idxs();
        let y_vals = y.vals();
        let mut dot = 0f32;
        let mut x_norm = 0f32;
        let mut y_norm = 0f32;
        while i < x_idxs.len() && j < y_idxs.len() {
            match x_idxs[i].cmp(&y_idxs[j]) {
                Ordering::Less => {
                    if self.0 { x_norm += x_vals[i].powi(2); }
                    i += 1;
                },
                Ordering::Greater => {
                    if self.0 { y_norm += y_vals[j].powi(2); }
                    j += 1
                },
                _ => {
                    dot += x_vals[i] * y_vals[j];
                    if self.0 { 
                        x_norm += x_vals[i].powi(2); 
                        y_norm += y_vals[j].powi(2); 
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        if self.0 {
            let xy_norm = x_norm.sqrt() * y_norm.sqrt();
            dot /= if xy_norm > 0. { xy_norm } else { 1f32 };
        } 
        1f32 - dot
    }
}

/*
pub struct Manhattan;

impl Distance<Sparse> for SparseManhattanDistance {
    fn distance(&self, x: &Sparse, y: &Sparse) -> f32 {
        let mut i = 0usize;
        let mut j = 0usize;
        let mut sum = 0f32;
        while i < x.1.len() && j < y.1.len() {
            match x.1[i].cmp(&y.1[j]) {
                Ordering::Less => {
                    sum += x.2[i].abs();
                    i += 1;
                },
                Ordering::Greater => {
                    sum += y.2[j].abs();
                    j += 1
                },
                _ => {
                    sum += (x.2[i] - y.2[j]).abs();
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    struct Sparse(Vec<usize>, Vec<f32>);

    impl SparseVec<f32> for Sparse {
        fn idxs(&self) -> &[usize] { &self.0 }
        fn vals(&self) -> &[f32] { &self.1 }
    }

    #[test]
    fn euclidean() {
        let d  = Euclidean;
        let x = vec![1f32, 2f32];
        let y = vec![0f32, 12f32];
        assert_eq!(d.distance(&x, &y), 101f32);
    }

    #[test]
    fn euclidean_sparse() {
        let d  = Euclidean;
        let x = Sparse(vec![0,1], vec![1f32, 2f32]);
        let y = Sparse(vec![1], vec![12f32]);
        assert_eq!(d.distance(&x, &y), 101f32);
    }

    #[test]
    fn cosine() {
        let d = Cosine(true);
        let x = vec![1f32, 2f32];
        let y = vec![0f32, 12f32];
        let denom = (1f32 + 4f32).sqrt() * 12f32;
        let c = (12f32 * 2f32) / denom;
        assert_eq!(d.distance(&x, &y), 1f32 - c);
    }

    #[test]
    fn cosine_sparse() {
        let d  = Cosine(true);
        let x = Sparse(vec![0,1], vec![1f32, 2f32]);
        let y = Sparse(vec![1], vec![12f32]);
        let denom = (1f32 + 4f32).sqrt() * 12f32;
        let c = (12f32 * 2f32) / denom;
        assert_eq!(d.distance(&x, &y), 1f32 - c);
    }

}
