use std::hash::Hash;
use std::collections::HashMap;
use std::marker::PhantomData;

use super::Evaluator;

#[derive(Clone)]
pub struct UniformEvaluator<K>(PhantomData<K>);

impl <K> UniformEvaluator<K> {
    pub fn new() -> Self { UniformEvaluator(PhantomData) }
}

impl <K: Eq + Hash + Clone> Evaluator<K,K> for UniformEvaluator<K> {
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

