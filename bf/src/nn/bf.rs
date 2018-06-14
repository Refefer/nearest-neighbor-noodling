extern crate rand;

use std::rc::Rc;

use self::rand::{XorShiftRng,SeedableRng,Rng};

use super::*;

struct Node<DT,LT> {
    point: Rc<Record<DT,LT>>,
    children: Vec<Node<DT,LT>>
}

impl <DT,LT> Node<DT,LT> {
    fn new(record: Rc<Record<DT, LT>>) -> Self {
        Node { point: record, children: Vec::new() }
    }
}

type RN<DT,LT> = Node<DT,LT>;

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
    mut node: &'a mut RN<DT,LT>,
    record: &Rc<Record<DT,LT>>
) -> &'a mut RN<DT,LT> {
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
    mut node: &'a RN<DT,LT>,
    x: &DT
) -> (f32, &'a RN<DT,LT>) {
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

    fn new(p: Rc<Record<DT,LT>>, k: usize, d: RBD<DT>, ld: RBLD<LT>) -> Self {
        BoundaryTree {
            root: Node::new(p),
            k: k,
            dist: d,
            lab_dist: ld
        }
    }
    
    fn insert(&mut self, record: Rc<Record<DT,LT>>) -> bool {
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

impl <DT,LT,Res> BoundaryForest<DT,LT,Res> {

    pub fn new<D: 'static + Distance<DT>, 
               L: 'static + LabelDistance<LT>, 
               E: 'static + Evaluator<LT,Res>>
        (mut points: Vec<Rc<Record<DT,LT>>>, k: usize, d: D, ld: L, ev: E)
    -> Self {
        let rd: RBD<DT> = Rc::new(Box::new(d));
        let rld: RBLD<LT> = Rc::new(Box::new(ld));

        // We instantiate each tree with a unique point
        let mut trees: Vec<_> = points.iter().map(|p| {
            BoundaryTree::new(p.clone(), k, rd.clone(), rld.clone())
        }).collect();

        // Shuffle remaining points and add them to each tree
        let mut prng = XorShiftRng::from_seed([2018,6,13,0]);
        for t in trees.iter_mut() {
            prng.shuffle(&mut points);
            for p in points.iter() {
                t.insert(p.clone());
            }
        }

        BoundaryForest {
            trees: trees,
            ev: Box::new(ev)
        }
    }

    pub fn insert(&mut self, x: DT, y: LT) -> () {
        let r = Rc::new(Record::new(x, y));
        for t in self.trees.iter_mut() {
            t.insert(r.clone());
        }
    }

    pub fn query(&self, x: &DT) -> Res {
        let v: Vec<_> = self.trees.iter().map(|t| t.query(x)).collect();
        self.ev.merge(v)
    }

}
