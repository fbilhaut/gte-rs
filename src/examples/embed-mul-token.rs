//! Complete example for text embedding using `gte-multilingual` (first token)

mod util;

fn main() -> gte::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gte-multilingual-base/tokenizer.json";
    const MODEL_PATH: &str = "models/gte-multilingual-base/onnx/model.onnx";

    let params = gte::params::Parameters::default().with_output_id("token_embeddings");
    let pipeline = gte::embed::pipeline::TextEmbeddingPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;
            
    let inputs = gte::embed::input::TextInput::from_str(&[
        "What is the capital of France?", 
        "How to implement quick sort in python?", 
        "Die Hauptstadt von Frankreich ist Paris.",
        "La capital de Francia es París.",
        "London is the capital of the UK",
    ]);

    let outputs = model.inference(inputs, &pipeline, &params)?;
    let distances = util::distances_to_first(&outputs);

    println!("Distances: {:?}", distances);     

    Ok(())
}
