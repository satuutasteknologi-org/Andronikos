use burn::optim::{AdamWConfig, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn_ndarray::NdArray;
use burn::backend::Autodiff;

type B = Autodiff<NdArray<f32>>;

fn test_optimizer() {
    let opt_config = AdamWConfig::new();
    let optimizer = opt_config.init::<B>(); // Wait, does it take B?
}
