use ort::session::SessionOutputs;

/// Output tensors right from the model
pub struct OutputTensors<'a> {
    pub outputs: SessionOutputs<'a, 'a>,
}

impl<'a> TryFrom<(SessionOutputs<'a, 'a>, ())> for OutputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_from(value: (SessionOutputs<'a, 'a>, ())) -> Result<Self, Self::Error> {
        Ok(OutputTensors { outputs: value.0 })
    }
}
