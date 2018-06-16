extern crate float_ord;

use std::mem::swap;
use std::collections::BinaryHeap;

use self::float_ord::FloatOrd;

use super::{Record,Distance,Evaluator};

pub struct KNN<DT,LT,Res> {
    k: usize,
    x: Vec<Record<DT,LT>>,
    dist: Box<Distance<DT>>,
    ev: Box<Evaluator<LT,Res>>
}

impl <DT,LT,Res> KNN<DT,LT,Res> {
    pub fn new< D: 'static + Distance<DT>, E: 'static + Evaluator<LT,Res>>(
        k: usize, 
        d: D, 
        ev: E
    ) -> Self {
        KNN { 
            k: k, 
            x: Vec::new(), 
            dist: Box::new(d),
            ev: Box::new(ev)
        }
    }

    pub fn len(&self) -> usize { self.x.len() }

    pub fn insert(&mut self, record: Record<DT,LT>) -> () {
        self.x.push(record)
    }

    pub fn query(&self, x: &DT) -> Option<Res> {
        if self.x.is_empty() { return None }

        let mut bh = BinaryHeap::new();
        for (i, xi) in self.x.iter().enumerate() {
            let dist = self.dist.distance(&xi.x, &x);
            bh.push((FloatOrd(dist), i));
            if i > self.k - 1 {
                bh.pop();
            }
        }
        let v = bh.drain().map(|x| ((x.0).0, &self.x[x.1].y)).collect();
        Some(self.ev.merge(v))
    }

    pub fn drain(&mut self) -> Vec<Record<DT,LT>> {
        let mut tmp = Vec::new();
        swap(&mut tmp, &mut self.x);
        tmp
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use distance::Euclidean;
    use evaluator::Uniform;
    #[test]
    fn simple_test() {
        let d  = Euclidean;
        let ev = Uniform::new();
        let mut nn = KNN::new(3, d, ev);

        assert_eq!(nn.query(&vec![-1f32, -2.1f32]), None);

        nn.insert(Record::new(vec![-2f32, -1f32], 1usize));
        nn.insert(Record::new(vec![-1f32, -2f32], 1usize));
        nn.insert(Record::new(vec![2f32, 1f32], 0usize));
        nn.insert(Record::new(vec![0f32, 0f32], 0usize));

        assert_eq!(nn.query(&vec![-1f32, -2.1f32]), Some(1usize));
        assert_eq!(nn.query(&vec![0.5f32, -0.1f32]), Some(0usize));
    }
}
