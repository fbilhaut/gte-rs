//! Parameters for embedding or re-ranking

use crate::embed::output::{ExtractorMode, OutputId};

/// Parameters
/// 
/// Prefer using `default()` and set individual parameters as needed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameters {
    max_length: Option<usize>,
    sigmoid: bool,
    token_types: bool,
    output_id: OutputId,
    mode: ExtractorMode,
}

impl Parameters {
    
    /// Input truncation (nb of tokens)
    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length = max_length;
        self
    }

    /// Input truncation (nb of tokens)
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    /// Apply sigmoid (for re-reanking)
    pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self
    }

    /// Apply sigmoid (for re-reanking)
    pub fn sigmoid(&self) -> bool {
        self.sigmoid
    }

    /// Does the model need `token_type_ids`
    pub fn with_token_types(mut self, b: bool) -> Self {
        self.token_types = b;
        self
    }

    /// Does the model need `token_type_ids`
    pub fn token_types(&self) -> bool {
        self.token_types
    }

    /// Set output tensor identifier (eg. `last_hidden_state`)
    pub fn with_output_id(mut self, id: &str) -> Self {
        self.output_id = id.into();
        self
    }

    /// Output tensor identifier
    pub fn output_id(&self) -> &OutputId {
        &self.output_id
    }

    /// Set embeddings extraction mode
    pub fn with_mode(mut self, mode: ExtractorMode) -> Self {
        self.mode = mode;
        self
    }

    /// Embeddings extraction mode
    pub fn mode(&self) -> ExtractorMode {
        self.mode
    }
    
}


impl Default for Parameters {
    fn default() -> Self {
        Self { 
            max_length: None,
            sigmoid: false,
            token_types: false,
            output_id: OutputId::default(),
            mode: ExtractorMode::default(),
        }
    }
}