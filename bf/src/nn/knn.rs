use super::{Record,RBD};

use std::rc::Rc;

pub struct KNN<DT,LT> {
    x: Vec<Rc<Record<DT,LT>>>,
    dist: RBD<DT>
}

impl <DT,LT> KNN<DT,LT> {
    pub fn new(d: RBD<DT>) -> Self {
        KNN { x: Vec::new(), dist: d }
    }

    pub fn len(&self) -> usize { self.x.len() }

    pub fn insert(&mut self, record: Rc<Record<DT,LT>>) -> () {
        self.x.push(record)
    }

    pub fn query(&self, x: &DT) -> Option<(f32, &LT)> {
        self.x.iter()
            .map(|xi| (self.dist.distance(&xi.x, &x), &xi.y))
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
    }
}

