//! Complete example for re-ranking
//! 
//! Reproduces the example from <https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base>
//! and checks for consistency of the results.
fn main() -> gte::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gte-modernbert-base/tokenizer.json";
    const MODEL_PATH: &str = "models/gte-reranker-modernbert-base/onnx/model.onnx";
    const APPLY_SIGMOID: bool = true;

    let params = gte::params::Parameters::default().with_sigmoid(APPLY_SIGMOID);
    let pipeline = gte::rerank::pipeline::RerankingPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;

    let inputs = gte::rerank::input::TextInput::from_str(&[
        ("what is the capital of China?", "Beijing"),
        ("how to implement quick sort in python?", "Introduction of quick sort"),
        ("how to implement quick sort in python?", "The weather is nice today"),
    ]);

    const EXPECTED_SIMILARITIES: [f32; 3] = if APPLY_SIGMOID { [0.8945664, 0.9213594, 0.15742092] } else { [2.1387, 2.4609, -1.6729] };    
    const EPSILON: f32 = if APPLY_SIGMOID { 0.00001 } else { 0.01 };

    let outputs = model.inference(inputs, &pipeline, &params)?;

    println!("Scores: {:?}", outputs.scores);
    assert!(gte::util::test::is_close_to_a(&outputs.scores.view(), &EXPECTED_SIMILARITIES, EPSILON));

    Ok(())
}
