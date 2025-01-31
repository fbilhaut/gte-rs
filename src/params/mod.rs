//! Parameters for embedding or re-ranking

/// Parameters
/// 
/// Prefer using `default()` and set individual parameters as needed.
pub struct Parameters {
    max_length: Option<usize>,
    sigmoid: bool,
}

impl Parameters {
    pub fn new(max_length: Option<usize>, sigmoid: bool) -> Self {
        Self { max_length, sigmoid }
    }

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
}

impl Default for Parameters {
    fn default() -> Self {
        Self { 
            max_length: None,
            sigmoid: false,
        }
    }
}