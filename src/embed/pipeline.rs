use std::path::Path;
use orp::pipeline::{Pipeline, PreProcessor, PostProcessor};
use crate::embed::output::EmbeddingsExtractor;
use crate::params::Parameters;
use crate::tokenizer::Tokenizer;
use crate::util::result::Result;

/// Pipeline for text embedding
pub struct TextEmbeddingPipeline {
    tokenizer: Tokenizer,
}


impl TextEmbeddingPipeline {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P, params: &Parameters) -> Result<Self> {
        Ok(Self { 
            tokenizer: Tokenizer::new(tokenizer_path, params.max_length())?
        })
    }
}


impl<'a> Pipeline<'a> for TextEmbeddingPipeline {
    type Input = super::input::TextInput;
    type Output = super::output::TextEmbeddings;
    type Context = ();
    type Parameters = Parameters;

    fn pre_processor(&self, _params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composable::composed![
            crate::commons::input::encoded::TextInputEncoder::new(&self.tokenizer),
            crate::commons::input::tensors::InputTensors::try_from,
            crate::commons::input::tensors::InputTensors::try_into            
        ]
    }

    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composable::composed![
            crate::commons::output::tensors::OutputTensors::try_from,
            EmbeddingsExtractor::new(params.output_id(), params.mode())
        ]
    }
}