use crate::util::result::Result;
use crate::commons::output::tensors::OutputTensors;

const TENSOR_LOGITS: &str = "logits";

/// Re-ranking output
pub struct TextSimilarities {
    pub scores: ndarray::Array1<f32>,
}


impl TextSimilarities {
    pub fn try_from(tensors: OutputTensors, sigmoid: bool) -> Result<Self> {
        // extract last hidden state from `ort` output
        let last_hidden_state = tensors.outputs.get(TENSOR_LOGITS).ok_or_else(|| format!("expected tensor not found in model output: {TENSOR_LOGITS}"))?;
        let scores = last_hidden_state.try_extract_tensor::<f32>()?;
        // reduce superfluous dimensionality 
        let scores = scores.slice(ndarray::s!(.., 0));
        // apply sigmoid        
        let scores = if sigmoid { crate::util::math::sigmoid_a(&scores) } else { scores.into_owned() };
        // job's done
        Ok(Self {  scores })
    }
}
