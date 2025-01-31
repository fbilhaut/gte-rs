use ndarray::*;

pub fn cosine_similarity<T: NdFloat>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> T {
    assert_eq!(a.len(), b.len());
    let dot_product = Zip::from(a).and(b).fold(T::zero(), |acc, &x, &y| acc + x * y);
    let magnitude_a = a.mapv(|x| x * x).sum().sqrt();
    let magnitude_b = b.mapv(|x| x * x).sum().sqrt();
    assert!(!(magnitude_a.is_zero() || magnitude_b.is_zero()));
    dot_product / (magnitude_a * magnitude_b)
}


pub fn softmax<T: NdFloat>(array: &ArrayView1<T>) -> Array1<T> {
    let exp_values: Array1<T> = array.mapv(|x| x.exp());
    let sum = exp_values.sum();
    exp_values / sum
}


pub fn sigmoid<T: NdFloat>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}


pub fn sigmoid_a<T: NdFloat>(array: &ArrayView1<T>) -> Array1<T> {
    array.mapv(sigmoid)
}

