/// Input for text embedding
pub struct TextInput {
    pub texts: Vec<String>,
}

impl TextInput {
    pub fn new(texts: Vec<String>) -> Self {
        Self { texts }
    }

    pub fn from_str(texts: &[&str]) -> Self {
        Self::new(
            texts.iter().map(|s| s.to_string()).collect(),
        )
    }
}

impl<'a> crate::commons::input::text::TextInput<'a> for TextInput {
    type InputType = String;

    fn into_encode_input(self) -> Vec<Self::InputType> {
        self.texts
    }
}


