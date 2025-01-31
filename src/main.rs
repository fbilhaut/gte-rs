//! Original tests, before re-implementing using `piport`

#![allow(dead_code)]

pub mod tokenizer;
pub mod embed;
pub mod util;
pub mod params;
pub mod commons;

use ndarray::ArrayView;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

fn main() -> util::result::Result<()> {
    test_embeddings()?;
    test_reranking()?;
    Ok(())
}


fn test_embeddings() -> util::result::Result<()> {
    
    let inputs = vec!["what is the capital of China?", "how to implement quick sort in python?", "Beijing", "sorting algorithms"];

    let mut truncation = TruncationParams::default();
    truncation.max_length = 128;

    let mut padding = PaddingParams::default();
    padding.strategy = PaddingStrategy::BatchLongest;    

    let mut tokenizer = tokenizers::Tokenizer::from_file("models/gte-modernbert-base/tokenizer.json")?;
    let tokenizer = tokenizer.with_padding(Some(padding)).with_truncation(Some(truncation))?;
    
    let encoded_seqs = tokenizer.encode_batch(inputs, true)?;
    let max_tokens = encoded_seqs.first().unwrap().len();
    let num_seq = encoded_seqs.len();

    let mut input_ids = ndarray::Array2::zeros((0, max_tokens));
    let mut attn_masks = ndarray::Array2::zeros((0, max_tokens));
    for tokens in encoded_seqs {
        input_ids.push_row(ArrayView::from(&to_i64(tokens.get_ids()).to_vec()))?;
        attn_masks.push_row(ArrayView::from(&to_i64(tokens.get_attention_mask()).to_vec()))?;
    }

    //println!("INPUTS:\n{input_ids}\n{attn_masks}\n");

    ////////////////////////////////////////////////////
    
    let inputs = ort::inputs!{
        "input_ids" => input_ids,
        "attention_mask" => attn_masks,    
    }?;

    let session = Session::builder()?   
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/gte-modernbert-base/onnx/model.onnx")?;


    let outputs = session.run(inputs)?;
    let outputs = outputs.get("last_hidden_state").ok_or("'last_hidden_state' tensor not found")?;
    let outputs = outputs.try_extract_tensor::<f32>()?;

    //println!("OUTPUTS:\n{outputs}");

    ////////////////////////////////////////////////////
    
    let expected = vec![0.4289073944091797, 0.7130911254882812, 0.33664554595947266];

    let first = outputs.slice(ndarray::s![0, 0, ..]);
    for i in 1..num_seq {
        let other = outputs.slice(ndarray::s![i, 0, ..]);
        let sim = crate::util::math::cosine_similarity(&first, &other);
        println!("{i} => {}%",  sim * 100.0);
        assert!(crate::util::test::is_close_to(sim, expected[i-1], 0.000001));
    }

    Ok(())
}


fn test_reranking() -> util::result::Result<()> {
    
    let inputs = vec![
        ("what is the capital of China?", "Beijing"),
        ("how to implement quick sort in python?", "Introduction of quick sort"),
        ("how to implement quick sort in python?", "The weather is nice today"),
    ];

    let mut truncation = TruncationParams::default();
    truncation.max_length = 128;

    let mut padding = PaddingParams::default();
    padding.strategy = PaddingStrategy::BatchLongest;    

    let mut tokenizer = tokenizers::Tokenizer::from_file("models/gte-reranker-modernbert-base/tokenizer.json")?;
    let tokenizer = tokenizer.with_padding(Some(padding)).with_truncation(Some(truncation))?;
    
    let encoded_seqs = tokenizer.encode_batch(inputs, true)?;
    let max_tokens = encoded_seqs.first().unwrap().len();

    let mut input_ids = ndarray::Array2::zeros((0, max_tokens));
    let mut attn_masks = ndarray::Array2::zeros((0, max_tokens));
    for tokens in encoded_seqs {
        input_ids.push_row(ArrayView::from(&to_i64(tokens.get_ids()).to_vec()))?;
        attn_masks.push_row(ArrayView::from(&to_i64(tokens.get_attention_mask()).to_vec()))?;
    }

    //println!("INPUTS:\n{input_ids}\n{attn_masks}\n");

    ////////////////////////////////////////////////////
    
    let inputs = ort::inputs!{
        "input_ids" => input_ids,
        "attention_mask" => attn_masks,    
    }?;

    let session = Session::builder()?   
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/gte-reranker-modernbert-base/onnx/model.onnx")?;


    let outputs = session.run(inputs)?;
    let outputs = outputs.get("logits").ok_or("'logits' tensor not found")?;
    let outputs = outputs.try_extract_tensor::<f32>()?;
    let outputs = outputs.into_dimensionality::<ndarray::Ix2>()?;
    let outputs_raw = outputs.slice(ndarray::s!(.., 0));
    //let outputs_probs = softmax(&outputs); // note: the comment in the example is not correct, this is not softmax but sigmoid !!
    let outputs_probs = util::math::sigmoid_a(&outputs_raw);

    println!("OUTPUTS:\nRaw: {outputs_raw}\nProb: {outputs_probs}");

    ////////////////////////////////////////////////////
    
    let expected = [ 2.1387, 2.4609, -1.6729];    
    for (i, expected) in expected.iter().enumerate() {
        assert!(util::test::is_close_to(*outputs_raw.get(i).unwrap(), *expected, 0.01));        
    }

    let expected_prob = [ 0.8945664, 0.9213594, 0.15742092 ];
    for (i, expected) in expected_prob.iter().enumerate() {
        assert!(util::test::is_close_to(*outputs_probs.get(i).unwrap(), *expected, 0.00001));        
    }

    Ok(())
}



fn to_i64(array: &[u32]) -> Vec<i64> {
    array.iter().into_iter().map(|x| *x as i64).collect()
}
