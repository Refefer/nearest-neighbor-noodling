use super::bf::BoundaryForest;
use super::{Distance,LabelDistance,Evaluator,Record};
use super::knn::KNN;

enum Model<DT,LT,Res> {
    Knn(KNN<DT,LT,Res>),
    BF(BoundaryForest<DT,LT,Res>)
}

pub struct Ann<DT,LT,Res> {
    model: Model<DT, LT, Res>,
    n_trees: usize,
    builder: Box<Fn(Vec<Record<DT,LT>>) -> BoundaryForest<DT,LT,Res>>
}

impl <DT,LT,Res> Ann<DT,LT,Res> {
    pub fn new<D: 'static + Distance<DT> + Clone, 
               L: 'static + LabelDistance<LT> + Clone, 
               E: 'static + Evaluator<LT,Res> + Clone> (
        n_trees: usize, 
        nn: usize, 
        k: usize, 
        dist: D, 
        ldist: L, 
        ev: E
    ) -> Self {
        Ann {
            model: Model::Knn(KNN::new(nn, dist.clone(), ev.clone())),
            n_trees: n_trees,
            builder: Box::new(
                move |points| BoundaryForest::new(points, k, dist.clone(), ldist.clone(), ev.clone())
            )
        }
    }

    pub fn insert(&mut self, record: Record<DT,LT>) -> () {
        let check = match self.model {
            Model::BF(ref mut model) => {
                model.insert(record);
                None
            },
            Model::Knn(ref mut model) => {
                model.insert(record);
                Some(model.len())
            }
        };

        // Check if we need to upgrade
        if let Some(size) = check {
            if size == self.n_trees {
                let points = match self.model {
                    Model::Knn(ref mut m) => Some(m.drain()),
                    _ => None
                };
                self.model = Model::BF((self.builder)(points.unwrap()))
            }
        }
    }

    pub fn query(&self, x: &DT) -> Option<Res> {
       match self.model {
            Model::BF(ref model) => {
                Some(model.query(&x))
            },
            Model::Knn(ref model) => {
                model.query(&x)
            }
        } 
    }

    pub fn load<I:Iterator<Item=Record<DT,LT>>>(&mut self, it: I) -> () {
        for item in it {
            self.insert(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use distance::Euclidean;
    use label_distance::OneOfK;
    use evaluator::Uniform;
    #[test]
    fn simple_test() {
        let d  = Euclidean;
        let ld  = OneOfK::new();
        let ev = Uniform::new();

        // Training
        let r1 =  Record::new(vec![-2f32, -1f32], 1usize);
        let r2 =  Record::new(vec![2f32, 1f32], 0usize);
        let r3 =  Record::new(vec![-1f32, -2f32], 1usize);
        let r4 =  Record::new(vec![0f32, 0f32], 0usize);

        let mut model = Ann::new(3, 1, 2, d, ld, ev);
        model.insert(r1);
        model.insert(r2);

        assert_eq!(model.query(&vec![-1f32, -2.1f32]), Some(1usize));
        assert_eq!(model.query(&vec![0.5f32, -0.1f32]), Some(0usize));

        model.insert(r3);
        model.insert(r4);
        assert_eq!(model.query(&vec![-1f32, -2.1f32]), Some(1usize));

        let x = vec![0f32, 0f32];
        let r = Record::new(x.clone(), 10usize);
        model.insert(r);

        assert_eq!(model.query(&x), Some(10usize));
    }
}
