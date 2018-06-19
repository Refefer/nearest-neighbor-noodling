extern crate itertools;
extern crate rayon;
extern crate rand;

use std::collections::HashSet;
use std::sync::Arc;

use self::rayon::prelude::*;
use self::rand::{XorShiftRng,SeedableRng,Rng};

use super::*;

struct Node<DT,LT> {
    point: Arc<Record<DT,LT>>,
    children: Vec<Node<DT,LT>>
}

impl <DT,LT> Node<DT,LT> {
    fn new(record: Arc<Record<DT, LT>>) -> Self {
        Node { point: record, children: Vec::new() }
    }
}

struct BoundaryTree<DT, LT>
{
    root: Node<DT, LT>,
    k: usize,
    dist: RBD<DT>,
    lab_dist: RBLD<LT>,
}

fn find_mut<'a, DT,LT>(
    dist: &RBD<DT>,
    k: usize,
    mut node: &'a mut Node<DT,LT>,
    record: &Arc<Record<DT,LT>>
) -> &'a mut Node<DT,LT> {
    let mut score = dist.distance(&node.point.x, &record.x);
    loop {
        if node.children.len() == 0 { return node }

        let (c_score, c_idx) = {
            node.children.iter().enumerate().map(|en| {
                    (dist.distance(&en.1.point.x, &record.x), en.0)
                })
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .unwrap()
        };

        // Check against root node when less than k
        if node.children.len() < k && score < c_score {
            return node
        }

        score = c_score;
        node = &mut {node}.children[c_idx];
    }
}

// Basically the same as find_mut, except immutable. it's a bummer
fn query<'a,DT,LT>(
    dist: &RBD<DT>,
    k: usize,
    mut node: &'a Node<DT,LT>,
    x: &DT
) -> (f32, &'a Node<DT,LT>) {
    let mut score = dist.distance(&node.point.x, &x);
    loop {
        if node.children.len() == 0 { return (score, node) }

        let (c_score, c_node) = {
            node.children.iter().map(|n| {
                    (dist.distance(&n.point.x, &x), n)
                })
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .unwrap()
        };

        // Check against root node when less than k
        if node.children.len() < k && score < c_score {
            return (score, node)
        }

        score = c_score;
        node = &c_node
    }
 
}


impl <DT,LT> BoundaryTree<DT,LT> {

    fn new(p: Arc<Record<DT,LT>>, k: usize, d: RBD<DT>, ld: RBLD<LT>) -> Self {
        BoundaryTree {
            root: Node::new(p),
            k: k,
            dist: d,
            lab_dist: ld
        }
    }
    
    fn insert(&mut self, record: Arc<Record<DT,LT>>) -> bool {
        let new_node = find_mut(&self.dist, self.k, &mut self.root, &record);
        if !self.lab_dist.equivalent(&new_node.point.y, &record.y) {
            new_node.children.push(Node::new(record));
            true
        } else {
            false
        }
    }

    fn query(&self, x: &DT) -> (f32, &LT) {
        let (s, rn) = query(&self.dist, self.k, &self.root, x);
        (s, &rn.point.y)
    }
}

pub struct BoundaryForest<DT,LT,Res> {
    trees: Vec<BoundaryTree<DT,LT>>,
    ev: Box<Evaluator<LT,Res>>
}

impl <DT: Send + Sync,LT: Send + Sync,Res> BoundaryForest<DT,LT,Res> {

    pub fn new<D: 'static + Distance<DT>, 
               L: 'static + LabelDistance<LT>, 
               E: 'static + Evaluator<LT,Res>>
        (points: Vec<Record<DT,LT>>, k: usize, d: D, ld: L, ev: E)
    -> Self {
        let rd: RBD<DT> = Arc::new(Box::new(d));
        let rld: RBLD<LT> = Arc::new(Box::new(ld));

        let mut ref_points: Vec<_> = points.into_iter().map(|p| Arc::new(p)).collect();

        // We instantiate each tree with a unique point
        let mut trees: Vec<_> = ref_points.iter().map(|p| {
            BoundaryTree::new(p.clone(), k, rd.clone(), rld.clone())
        }).collect();

        // Shuffle remaining points and add them to each tree
        let mut prng = XorShiftRng::from_seed([2018,6,13,0]);
        for t in trees.iter_mut() {
            prng.shuffle(&mut ref_points);
            for p in ref_points.iter() {
                t.insert(p.clone());
            }
        }

        BoundaryForest {
            trees: trees,
            ev: Box::new(ev)
        }
    }

    pub fn insert(&mut self, r: Record<DT,LT>) -> bool {
        let rr = Arc::new(r);
        self.trees.par_iter_mut()
            .fold(|| true, |acc,t| { acc && t.insert(rr.clone()) })
            .reduce(|| true, |a,b| a && b)
    }

    pub fn batch_insert<I: Iterator<Item=Record<DT,LT>>>(&mut self, mut r: I, size: usize) -> usize {
        let mut pass = 0u32;
        let mut total_count = 0usize;
        loop {
            let batch: Vec<Arc<Record<DT,LT>>> = 
                (&mut r).take(size).map(|x| Arc::new(x)).collect();
            // Finished when we're out of items
            if batch.len() == 0 { break }
            pass += 1;
            total_count += self.trees.par_iter_mut()
                .enumerate()
                .fold(|| HashSet::new(), |mut acc,it| { 
                    let (i, t) = it;
                    let mut prng = XorShiftRng::from_seed([2018,6,pass,i as u32]);
                    let mut my_batch: Vec<_> = batch.iter().enumerate().collect();
                    prng.shuffle(&mut my_batch);
                    for (idx, p) in my_batch.into_iter() {
                        if t.insert(p.clone()) {
                            acc.insert(idx);
                        }
                    }
                    acc
                })
                .reduce(|| HashSet::new(), |mut a,b| {
                    for idx in b {
                        a.insert(idx);
                    }
                    a
                })
                .len();
        }
        total_count
    }

    pub fn query(&self, x: &DT) -> Res {
        let v: Vec<_> = self.trees.par_iter().map(|t| t.query(x)).collect();
        self.ev.merge(v)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use distance::Euclidean;
    use label_distance::OneOfK;
    use evaluator::Bagging;
    #[test]
    fn simple_test() {
        let d  = Euclidean;
        let ld  = OneOfK::new();
        let ev = Bagging::new();
        let v = vec![
            Record::new(vec![-2f32, -1f32], 1usize),
            Record::new(vec![-1f32, -2f32], 1usize),
            Record::new(vec![2f32, 1f32], 0usize),
            Record::new(vec![0f32, 0f32], 0usize)
        ];

        let mut model = BoundaryForest::new(v, 2, d, ld, ev);

        assert_eq!(model.query(&vec![-1f32, -2.1f32]), 1usize);
        assert_eq!(model.query(&vec![0.5f32, -0.1f32]), 0usize);

        let x = vec![0f32, 0f32];
        let r = Record::new(x.clone(), 10usize);
        model.insert(r);

        assert_eq!(model.query(&x), 10usize);
    }
}
