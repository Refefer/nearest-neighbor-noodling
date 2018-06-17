use std::hash::Hash;
use std::collections::HashSet;
use std::marker::PhantomData;
use super::LabelDistance;

#[derive(Clone)]
pub struct OneOfK<K>(PhantomData<K>);

impl <K> OneOfK<K> {
    pub fn new() -> Self { OneOfK(PhantomData) }
}

impl <K: Eq> LabelDistance<K> for OneOfK<K> {
    fn equivalent(&self, x: &K, y: &K) -> bool {
        x == y
    }
}


#[derive(Clone)]
pub struct Jaccard<K>(f64, PhantomData<K>);

impl <K> Jaccard<K> {
    pub fn new(threshold: f64) -> Self { Jaccard(threshold, PhantomData) }
}

impl <K: Eq + Hash> LabelDistance<HashSet<K>> for Jaccard<K> {
    fn equivalent(&self, x: &HashSet<K>, y: &HashSet<K>) -> bool {
        let num = x.intersection(y).collect::<HashSet<&K>>().len();
        let denom= x.union(y).collect::<HashSet<&K>>().len();
        (num as f64 / denom as f64) > self.0
    }
}


#[derive(Clone)]
pub struct ThresholdRegression(pub f32);

impl LabelDistance<f32> for ThresholdRegression {
    fn equivalent(&self, x: &f32, y: &f32) -> bool {
        (x - y).abs() < self.0
    }
}

#[derive(Clone)]
pub struct Index;

impl <K> LabelDistance<K> for Index {
    fn equivalent(&self, _x: &K, _y: &K) -> bool {
        false
    }
}


