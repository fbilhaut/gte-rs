//! Complete example for text embedding using Qwen3 embedding

fn main() -> gte::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/qwen3-embedding-0.6B/tokenizer.json";
    const MODEL_PATH: &str = "models/qwen3-embedding-0.6B/onnx/model.onnx";

    let params = gte::params::Parameters::default()
        .with_positions(true)
        .with_precision(gte::embed::output::Precision::F16)
        ;
    let pipeline = gte::embed::pipeline::TextEmbeddingPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;
            
    let inputs = gte::embed::input::TextInput::from_str(&[
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: what is the biggest city in China?", 
        "How to implement quick sort in python?", 
        "The capital of China is Beijing", 
        "Everything about sorting algorithms.",
    ]);

    //const EXPECTED_DISTANCES: [f32; 3] = [0.4289073944091797, 0.7130911254882812, 0.33664554595947266];
    //const EPSILON: f32 = 0.000001;

    let outputs = model.inference(inputs, &pipeline, &params)?;
    let distances = gte::util::test::distances_to_first(&outputs);

    println!("Distances: {:?}", distances); 
    //assert!(gte::util::test::is_close_to_a(&distances.view(), &EXPECTED_DISTANCES, EPSILON));

    Ok(())
}
