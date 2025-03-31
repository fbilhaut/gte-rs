use composable::Composable;
use crate::tokenizer::Tokenizer;

/// Encoded text input (using a tokenizer)
pub struct EncodedInput {
    pub input_ids: ndarray::Array2<i64>,
    pub attn_masks: ndarray::Array2<i64>,
    pub type_ids: Option<ndarray::Array2<i64>>,
}


pub struct TextInputEncoder<'a> {
    tokenizer: &'a Tokenizer,
}


impl<'a> TextInputEncoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self { tokenizer }
    }
}


impl<'a, T> Composable<T, EncodedInput> for TextInputEncoder<'a> where T: super::text::TextInput<'a> {
    fn apply(&self, input: T) -> composable::Result<EncodedInput> {
        let input = input.into_encode_input();
        let tokenized = self.tokenizer.tokenize(input)?;
        Ok(EncodedInput{
            input_ids: tokenized.input_ids,
            attn_masks: tokenized.attn_masks,
            type_ids: tokenized.type_ids,
        })
    }
}