extern crate rand;

pub mod bf;
pub mod knn;
pub mod ann;

pub mod distance;
pub mod evaluator;
pub mod label_distance;

use std::sync::Arc;

pub trait Distance<DataType>: Send + Sync {
    fn distance(&self, x: &DataType, y: &DataType) -> f32;
}

pub trait LabelDistance<LabelType>: Send + Sync {
    fn equivalent(&self, x: &LabelType, y: &LabelType) -> bool;
}

pub trait Evaluator<LabelType,Res> {
    fn merge(&self, scores: Vec<(f32, &LabelType)>) -> Res;
}

pub struct Record<DT,LT> { x: DT, y: LT }

impl <DT,LT> Record<DT,LT> {
    pub fn new(x: DT, y: LT) -> Self {
        Record {x: x, y: y}
    }
}

type RBD<DT> = Arc<Box<Distance<DT> + Send + Sync>>;
type RBLD<LT> = Arc<Box<LabelDistance<LT> + Send + Sync>>;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
