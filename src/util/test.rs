use ndarray::*;

/// For testing purposes
pub fn is_close_to<T: NdFloat>(v1: T, v2: T, epsilon: T) -> bool {
    (v1 - v2).abs() <= epsilon
}

/// For testing purposes
pub fn is_close_to_a<T: NdFloat>(a1: &ArrayView1<T>, a2: &[T], epsilon: T) -> bool {
    assert!(a1.len() == a2.len());
    for i in 0..a1.len() {
        if !is_close_to(*a1.get(i).unwrap(), a2[i], epsilon) {
            return false;
        }
    }
    true
}

/// For testing purposes: measures the distance between the first sequence and every subsequent one
pub fn distances_to_first(outputs: &crate::embed::output::TextEmbeddings) -> ndarray::Array1<f32> {
    assert!(outputs.len() >= 2);
    let first = outputs.embeddings(0);
    outputs.embeddings.rows().into_iter()
        .skip(1)
        .map(|e| crate::util::math::cosine_similarity(&first, &e))
        .collect()
}