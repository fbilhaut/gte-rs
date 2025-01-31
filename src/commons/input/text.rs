/// Input text abstraction, use `embed::input::TextInput` or `rerank::input::TextInput` implementations.
pub trait TextInput<'s> {
    type InputType: Into<tokenizers::EncodeInput<'s>> + Send;
    
    fn into_encode_input(self) -> Vec<Self::InputType>;
}
