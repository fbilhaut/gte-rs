use std::collections::HashMap;

use ort::session::SessionInputs;
use crate::util::result::Result;
use crate::commons::input::encoded::EncodedInput;

const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTN_MASKS: &str = "attention_mask";
const TENSOR_TOKEN_TYPE_IDS: &str = "token_type_ids";
const TENSOR_POSITION_IDS: &str = "position_ids";


/// Input tensors, ready for inferences
pub struct InputTensors<'a> {
    pub inputs: SessionInputs<'a, 'a>,
}


impl<'a> InputTensors<'a> {
    fn make_session_inputs(input: EncodedInput) -> Result<SessionInputs<'a, 'a>> {
        let mut inputs: HashMap<&str, ort::value::Value<ort::value::TensorValueType<i64>>> = HashMap::new();
        inputs.insert(TENSOR_INPUT_IDS, ort::value::Value::from_array(input.input_ids)?);
        inputs.insert(TENSOR_ATTN_MASKS, ort::value::Value::from_array(input.attn_masks)?);
        if let Some(type_ids) = input.type_ids {
            inputs.insert(TENSOR_TOKEN_TYPE_IDS, ort::value::Value::from_array(type_ids)?);
        }
        if let Some(position_ids) = input.position_ids {
            inputs.insert(TENSOR_POSITION_IDS, ort::value::Value::from_array(position_ids)?);
        } 
        Ok(SessionInputs::try_from(inputs)?)        
    }

    pub fn input_tensors(with_token_types: bool, with_position_ids: bool) -> Vec<&'static str> {
        let mut result = vec![TENSOR_INPUT_IDS, TENSOR_ATTN_MASKS];
        if with_token_types { result.push(TENSOR_TOKEN_TYPE_IDS); }
        if with_position_ids { result.push(TENSOR_POSITION_IDS); }
        result
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
