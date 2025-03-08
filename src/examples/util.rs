//! Utilities for the examples


/// Measures the distance between the first sequence and every subsequent one
pub fn distances_to_first(outputs: &gte::embed::output::TextEmbeddings) -> ndarray::Array1<f32> {
    assert!(outputs.len() >= 2);
    let first = outputs.embeddings(0);
    outputs.embeddings.rows().into_iter()
        .skip(1)
        .map(|e| gte::util::math::cosine_similarity(&first, &e))
        .collect()
}