use std::collections::HashSet;
use std::path::Path;
use orp::pipeline::{Pipeline, PreProcessor, PostProcessor};
use crate::params::Parameters;
use crate::tokenizer::Tokenizer;
use crate::util::result::Result;

/// Pipeline for re-ranking
pub struct RerankingPipeline {
    tokenizer: Tokenizer,
    expected_inputs: HashSet<&'static str>,
}


impl RerankingPipeline {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P, params: &Parameters) -> Result<Self> {
        Ok(Self { 
            tokenizer: Tokenizer::new(tokenizer_path, params.max_length(), params.token_types(), params.positions())?,
            expected_inputs: crate::commons::input::tensors::InputTensors::input_tensors(params.token_types(), params.positions()).into_iter().collect(),
        })
    }
}


impl<'a> Pipeline<'a> for RerankingPipeline {
    type Input = super::input::TextInput;
    type Output = super::output::TextSimilarities;
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
            |tensors| super::output::TextSimilarities::try_from(tensors, params.sigmoid())
        ]
    }

    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(self.expected_inputs.iter().copied())
    }

    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(std::iter::once(super::output::TextSimilarities::output_tensor()))
    }
}
