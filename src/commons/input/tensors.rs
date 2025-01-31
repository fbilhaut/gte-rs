use ort::session::SessionInputs;
use crate::util::result::Result;
use crate::commons::input::encoded::EncodedInput;

const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTN_MASKS: &str = "attention_mask";

/// Input tensors, ready for inferences
pub struct InputTensors<'a> {
    pub inputs: SessionInputs<'a, 'a>,    
}


impl<'a> TryFrom<EncodedInput> for InputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_from(input: EncodedInput) -> Result<Self> {
        Ok(Self {
            inputs: ort::inputs!{
                TENSOR_INPUT_IDS => input.input_ids,
                TENSOR_ATTN_MASKS => input.attn_masks,    
            }?.into()
        })
    }
}


impl<'a> TryInto<(SessionInputs<'a, 'a>, ())> for InputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_into(self) -> Result<(SessionInputs<'a, 'a>, ())> {
        Ok((self.inputs, ()))
    }    
}
