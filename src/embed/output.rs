use composable::Composable;
use crate::commons::output::tensors::OutputTensors;


/// Text embedding output
pub struct TextEmbeddings {
    pub embeddings: ndarray::Array2<f32>,
}

impl TextEmbeddings {
    pub fn embeddings(&self, index: usize) -> ndarray::ArrayView1<f32> {
        self.embeddings.slice(ndarray::s![index, ..])
    }

    pub fn len(&self) -> usize {
        self.embeddings.dim().0
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}


/// Output tensor identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputId(String);
impl Default for OutputId { fn default() -> Self { OutputId("last_hidden_state".into()) } }
impl From<String> for OutputId { fn from(s: String) -> Self { OutputId(s) } }
impl From<&str> for OutputId { fn from(s: &str) -> Self { OutputId(s.to_string()) } }
impl AsRef<str> for OutputId { fn as_ref(&self) -> &str { &self.0 } }
impl std::ops::Deref for OutputId { type Target = str; fn deref(&self) -> &Self::Target { &self.0 } }
impl std::fmt::Display for OutputId { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { self.0.fmt(f) } }


/// Defines the way embeddings are extracted from the output tensor
#[derive(Clone, Copy)]
pub enum ExtractorMode {
    /// The tensor is expected to provide directly usable embeddings for each sequence
    Raw,
    /// The tensor is expected to provide embeddings for each token, and we use one's vector as sequence embedding (usually the first one)
    Token(usize),
}

impl Default for ExtractorMode {
    fn default() -> Self { ExtractorMode::Token(0) }
}


/// Composable that extracts the embeddings from the output tensors, according to the specified output id and extraction mode
#[derive(Default)]
pub struct EmbeddingsExtractor {
    output_id: OutputId,
    mode: ExtractorMode,    
}

impl EmbeddingsExtractor {
    pub fn new(output_id: &OutputId, mode: ExtractorMode) -> Self {
        Self { 
            output_id: output_id.clone(), 
            mode,
        }
    }
}

impl Composable<OutputTensors<'_>, TextEmbeddings> for EmbeddingsExtractor {
    fn apply(&self, output_tensors: OutputTensors) -> composable::Result<TextEmbeddings> {
        // extract the tensor from the ORT output
        let output_tensor = output_tensors.outputs.get(&self.output_id).ok_or_else(|| format!("tensor not found in model output: {}", self.output_id))?;
        let output_tensor = output_tensor.try_extract_tensor::<f32>()?;
        
        // extract the actual embeddings depending on the desired mode
        match self.mode {
            ExtractorMode::Raw => {
                // the raw output tensor is supposed to provide the actual embeddings by sequence
                let embeddings = output_tensor.into_dimensionality::<ndarray::Ix2>()?;
                Ok(TextEmbeddings { embeddings: embeddings.into_owned() })
            },
            ExtractorMode::Token(index) => {
                // we select the selected token (by index) of each sequence         
                let embeddings = output_tensor.slice(ndarray::s![.., index, ..]);
                Ok(TextEmbeddings { embeddings: embeddings.into_owned() })
            },
        }
    }
}
