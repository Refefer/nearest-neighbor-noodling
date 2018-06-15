use std::marker::PhantomData;
use super::LabelDistance;

#[derive(Clone)]
pub struct OneOfKClassification<K>(PhantomData<K>);

impl <K> OneOfKClassification<K> {
    pub fn new() -> Self {
        OneOfKClassification(PhantomData)
    }
}

impl <K: Eq> LabelDistance<K> for OneOfKClassification<K> {
    fn equivalent(&self, x: &K, y: &K) -> bool {
        x == y
    }
}
