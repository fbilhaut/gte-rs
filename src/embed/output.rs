use crate::commons::output::tensors::OutputTensors;

const TENSOR_LAST_HIDDEN_STATE: &str = "last_hidden_state";

/// Text embedding output
pub struct TextEmbeddings {
    pub embeddings: ndarray::Array2<f32>,
}

impl TryFrom<OutputTensors<'_>> for TextEmbeddings {
    type Error = crate::util::result::Error;

    fn try_from(tensors: OutputTensors) -> std::result::Result<Self, Self::Error> {
       // extract last hidden state from ORT output
       let last_hidden_state = tensors.outputs.get(TENSOR_LAST_HIDDEN_STATE).ok_or_else(|| format!("tensor not found in model output: {TENSOR_LAST_HIDDEN_STATE}"))?;
       let embeddings = last_hidden_state.try_extract_tensor::<f32>()?;
       // select the first token of each sequence
       let embeddings = embeddings.slice(ndarray::s![.., 0, ..]);        
       // job's done
       Ok(Self { embeddings: embeddings.into_owned() })
    }
}

impl TextEmbeddings {
    pub fn embeddings(&self, index: usize) -> ndarray::ArrayView1<f32> {
        self.embeddings.slice(ndarray::s![index, ..])
    }

    pub fn len(&self) -> usize {
        self.embeddings.dim().0
    }    
}
