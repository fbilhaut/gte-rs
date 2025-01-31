/// Input for re-ranking
pub struct TextInput {
    pub pairs: Vec<(String, String)>,
}

impl TextInput {
    pub fn new(pairs: Vec<(String, String)>) -> Self {
        Self { pairs }
    }

    pub fn from_str(texts: &[(&str, &str)]) -> Self {
        Self::new(
            texts.iter().map(|(s, t)| (s.to_string(), t.to_string())).collect(),
        )
    }
}

impl<'a> crate::commons::input::text::TextInput<'a> for TextInput {
    type InputType = (String, String);

    fn into_encode_input(self) -> Vec<Self::InputType> {
        self.pairs
    }
}
