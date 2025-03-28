# 🧲 `gte-rs`: general text embedding and re-ranking in Rust

## 💬 Introduction

This crate provides simple pipelines that can be used out-of-the box to perform **text-embedding** and **re-ranking** using ONNX models.

They are built with [`🧩 orp`](https://github.com/fbilhaut/orp) (which relies on the [`🦀 ort`](https://ort.pyke.io) runtime), and use [`🤗 tokenizers`](https://github.com/huggingface/tokenizers) for token encoding.


## 🎓 Examples

```toml
[dependencies]
"gte-rs" = "0.9.1"
"orp" = "0.9.2"
```

**Embedding:**

```rust
let params = Parameters::default();
let pipeline = TextEmbeddingPipeline::new("gte-modernbert-base/tokenizer.json", &params)?;
let model = Model::new("gte-modernbert-base/model.onnx", RuntimeParameters::default())?;
            
let inputs = TextInput::from_str(&[
    "text content", 
    "some more content",
    //...
]);

let embeddings = model.inference(inputs, &pipeline, &params)?;
```

**Re-ranking:**

```rust
let params = Parameters::default();
let pipeline = RerankingPipeline::new("gte-reranker-modernbert-base/tokenizer.json", &params)?;
let model = Model::new("gte-reranker-modernbert-base/model.onnx", RuntimeParameters::default())?;

let inputs = TextInput::from_str(&[
    ("one candidate", "query"),
    ("another candidate", "query"),
    //...
]);

let similarities = model.inference(inputs, &pipeline, &params)?;
```

Please refer the the source code in `examples` for complete examples.


## 🧬 Models

### Alibaba's `gte-modernbert`

For english language, the [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) model outperforms larger models on retrieval with only 149M parameters, and runs efficiently on GPU and CPU. The [`gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) version does re-ranking with similar characteristics. This [post](https://www.linkedin.com/feed/update/urn:li:activity:7287831390425870336/) provides interesting insights about them.

### Other

This crate should be usable out-of-the box with other models, or easily adapted to other ones.
Please report your own tests or requirements!


## 👉 Related

This project follows the same principles as the ones below. Refer to their documentation for more details:

* 🌿 [gline-rs](https://github.com/fbilhaut/gline-rs): inference engine for GLiNER models
* 🏷️ [gliclass-rs](https://github.com/fbilhaut/gliclass-rs): inference engine for GLiClass models
