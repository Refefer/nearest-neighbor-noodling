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

pub struct ThresholdReg(f32);

impl LabelDistance<f32> for ThresholdReg{
    fn equivalent(&self, x: &f32, y: &f32) -> bool {
        (x - y).abs() < self.0
    }
}
