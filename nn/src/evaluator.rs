extern crate float_ord;

use std::hash::Hash;
use std::collections::HashMap;
use std::marker::PhantomData;

use self::float_ord::FloatOrd;

use super::Evaluator;

#[derive(Clone)]
pub struct Uniform<K>(PhantomData<K>);

impl <K> Uniform<K> {
    pub fn new() -> Self { Uniform(PhantomData) }
}

impl <K: Eq + Hash + Clone> Evaluator<K,K> for Uniform<K> {
    fn merge(&self, scores: Vec<(f32, &K)>) -> K {
        let mut hm = HashMap::new();
        for (_, k) in scores {
            let e = hm.entry(k).or_insert(0usize);
            *e += 1;
        }

        let (c, _) = hm.iter().max_by_key(|x| x.1).unwrap();
        (*c).clone()
    }
}

#[derive(Clone)]
pub struct UniformDF<K>(PhantomData<K>);

impl <K> UniformDF<K> {
    pub fn new() -> Self { UniformDF(PhantomData) }
}

impl <K: Eq + Hash + Clone> Evaluator<K,HashMap<K,usize>> for UniformDF<K> {
    fn merge(&self, scores: Vec<(f32, &K)>) -> HashMap<K,usize> {
        let mut hm = HashMap::new();
        for (_, k) in scores {
            let e = hm.entry(k.clone()).or_insert(0usize);
            *e += 1;
        }
        hm
    }
}

#[derive(Clone)]
pub struct ShepardClassifier<K>(PhantomData<K>);

impl <K> ShepardClassifier<K> {
    pub fn new() -> Self { ShepardClassifier(PhantomData) }
}

impl <K: Eq + Hash + Clone> Evaluator<K,K> for ShepardClassifier<K> {
    fn merge(&self, scores: Vec<(f32, &K)>) -> K {
        let mut hm = HashMap::new();
        for (dist, k) in scores {
            let e = hm.entry(k.clone()).or_insert(0f32);
            let k = 1. / (dist + 1e-6);
            *e += k;
        }
        let (k, _) = hm.iter()
            .max_by_key(|x| FloatOrd(*x.1))
            .unwrap();
        k.clone()
    }
}

#[derive(Clone)]
pub struct ShepardRegressor;

impl Evaluator<f32,f32> for ShepardRegressor {
    fn merge(&self, scores: Vec<(f32, &f32)>) -> f32 {
        let mut score = 0f32;
        let mut denom = 0f32;
        for (dist, k) in scores {
            denom += 1. / (dist + 1e-6);
            score += k / (dist + 1e-6) ;
        }
        score / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shep_reg() {
        let v = vec![(0.1, &2.), (0.3, &10.)];
        let num = 2. / (0.1 + 1e-6) + 10. / (0.3 + 1e-6);
        let denom = 1. / (0.1 + 1e-6) + 1. / (0.3 + 1e-6);
        let result = ShepardRegressor.merge(v);
        assert_eq!(result, num / denom);
    }
}
