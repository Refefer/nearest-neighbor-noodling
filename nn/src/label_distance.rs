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


