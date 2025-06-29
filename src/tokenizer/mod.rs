//! Wrapper around 🤗 tokenizers
use std::path::Path;
use crate::util::result::Result;

/// Wrapper around 🤗 tokenizers
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    with_type_ids: bool,
    with_position_ids: bool,
}

pub struct Tokenized {
    pub input_ids: ndarray::Array2<i64>, 
    pub attn_masks: ndarray::Array2<i64>,
    pub type_ids: Option<ndarray::Array2<i64>>,
    pub position_ids: Option<ndarray::Array2<i64>>,
}

impl Tokenizer {

    pub fn new<P: AsRef<Path>>(tokenizer_path: P, max_length: Option<usize>, with_type_ids: bool, with_position_ids: bool) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)?;
        
        if let Some(length) = max_length {
            let mut truncation = tokenizers::TruncationParams::default();
            truncation.max_length = length;
            tokenizer.with_truncation(Some(truncation))?;
        }

        let mut padding = tokenizers::PaddingParams::default();
        padding.strategy = tokenizers::PaddingStrategy::BatchLongest;    
        
        tokenizer.with_padding(Some(padding));
        
        Ok(Self { tokenizer, with_type_ids, with_position_ids })
    }

    pub fn tokenize<'s, E: Into<tokenizers::EncodeInput<'s>> + Send>(&self, input: Vec<E>) -> Result<Tokenized> {
        let encodings = self.tokenizer.encode_batch(input, true)?;
        let max_tokens = encodings.first().map(|x| x.len()).unwrap_or(0);
        let mut input_ids = ndarray::Array2::zeros((0, max_tokens));
        let mut attn_masks = ndarray::Array2::zeros((0, max_tokens));
        let mut type_ids = self.with_type_ids.then(|| ndarray::Array2::zeros((0, max_tokens)));
        let mut position_ids = self.with_position_ids.then(|| ndarray::Array2::zeros((0, max_tokens)));
        for encoding in encodings {
            input_ids.push_row(ndarray::ArrayView::from(&Self::to_i64(encoding.get_ids()).to_vec()))?;            
            attn_masks.push_row(ndarray::ArrayView::from(&Self::to_i64(encoding.get_attention_mask())))?;
            if let Some(type_ids) = type_ids.as_mut() {
                type_ids.push_row(ndarray::ArrayView::from(&Self::to_i64(encoding.get_type_ids())))?;
            }
            if let Some(position_ids) = position_ids.as_mut() {
                position_ids.push_row(ndarray::ArrayView::from(&Self::make_position_ids(&encoding)))?;
            }
        }
        Ok(Tokenized { input_ids, attn_masks, type_ids, position_ids })
    }

    fn to_i64(array: &[u32]) -> Vec<i64> {
        array.iter().map(|x| *x as i64).collect()
    }

    fn make_position_ids(encoding: &tokenizers::Encoding) -> Vec<i64> {
        encoding.get_attention_mask()
            .iter()
            .enumerate()
            .map(|(i, &mask)| if mask == 1 { i as i64 } else { 0 })
            .collect()
    }

}