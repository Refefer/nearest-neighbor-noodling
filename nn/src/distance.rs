use super::Distance;

#[derive(Clone)]
pub struct EucDistance;

impl Distance<Vec<f32>> for EucDistance {
    fn distance(&self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        x.iter().zip(y.iter()).map(|xyi| (xyi.0 - xyi.1).powi(2)).sum()
    }
}
