use ort::session::SessionInputs;
use crate::util::result::Result;
use crate::commons::input::encoded::EncodedInput;

const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTN_MASKS: &str = "attention_mask";
const TENSOR_TOKEN_TYPE_IDS: &str = "token_type_ids";


/// Input tensors, ready for inferences
pub struct InputTensors<'a> {
    pub inputs: SessionInputs<'a, 'a>,
}


impl<'a> InputTensors<'a> {
    fn make_session_inputs(input: EncodedInput) -> Result<SessionInputs<'a, 'a>> {
        if let Some(type_ids) = input.type_ids {
            Ok(ort::inputs! { TENSOR_INPUT_IDS => input.input_ids, TENSOR_ATTN_MASKS => input.attn_masks, TENSOR_TOKEN_TYPE_IDS => type_ids }?.into())
        }
        else {
            Ok(ort::inputs! { TENSOR_INPUT_IDS => input.input_ids, TENSOR_ATTN_MASKS => input.attn_masks }?.into())
        }        
    }

    pub fn input_tensors(with_token_types: bool) -> Vec<&'static str> {
        if with_token_types { vec![TENSOR_INPUT_IDS, TENSOR_ATTN_MASKS, TENSOR_TOKEN_TYPE_IDS] }
        else { vec![TENSOR_INPUT_IDS, TENSOR_ATTN_MASKS] }
    }
}


impl TryFrom<EncodedInput> for InputTensors<'_> {
    type Error = crate::util::result::Error;

    fn try_from(input: EncodedInput) -> Result<Self> {
        Ok(Self { inputs: Self::make_session_inputs(input)? })
    }
}


impl<'a> TryInto<(SessionInputs<'a, 'a>, ())> for InputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_into(self) -> Result<(SessionInputs<'a, 'a>, ())> {
        Ok((self.inputs, ()))
    }    
}
