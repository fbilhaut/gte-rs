//! Complete example for text embedding
//! 
//! Reproduces the example from <https://huggingface.co/Alibaba-NLP/gte-modernbert-base>
//! and checks for consistency of the results.

fn main() -> gte::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gte-modernbert-base/tokenizer.json";
    const MODEL_PATH: &str = "models/gte-modernbert-base/onnx/model.onnx";

    let params = gte::params::Parameters::default();
    let pipeline = gte::embed::pipeline::TextEmbeddingPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;
            
    let inputs = gte::embed::input::TextInput::from_str(&[
        "what is the capital of China?", 
        "how to implement quick sort in python?", 
        "Beijing", 
        "sorting algorithms",
    ]);

    const EXPECTED_DISTANCES: [f32; 3] = [0.4289073944091797, 0.7130911254882812, 0.33664554595947266];
    const EPSILON: f32 = 0.000001;

    let outputs = model.inference(inputs, &pipeline, &params)?;
    let distances = gte::util::test::distances_to_first(&outputs);

    println!("Distances: {:?}", distances); 
    assert!(gte::util::test::is_close_to_a(&distances.view(), &EXPECTED_DISTANCES, EPSILON));

    Ok(())
}
