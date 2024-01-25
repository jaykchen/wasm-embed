#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use once_cell::sync::OnceCell;
use serde::{ Deserialize, Serialize };
use wasi_nn::{ Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType };
use wasm_embed::*;
type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
use std::sync::Mutex;

const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";

static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();
static CTX_SIZE: OnceCell<usize> = OnceCell::new();
static GRAPH: OnceCell<Mutex<WasiNnGraph>> = OnceCell::new();
static METADATA: OnceCell<Metadata> = OnceCell::new();

// #[tokio::main(flavor = "current_thread")]
// async fn main() -> anyhow::Result<()> {
//     // let tokenizer = Tokenizer::from_file("path/to/tokenizer").unwrap();
//     // let tokens = tokenizer.encode("Your text here", true).unwrap();
//     // let token_ids = Tensor::new(tokens.get_ids(), device).unwrap().unsqueeze(0);
//     // let token_type_ids = token_ids.zeros_like().unwrap();

//     let metadata = Metadata {
//         log_enable: true,
//         ctx_size: 1024,
//         n_predict: 1,
//         n_gpu_layers: 0,
//         batch_size: 1,
//         temp: 0.7,
//         repeat_penalty: 1.0,
//         ..Default::default()
//     };

//     // load the model into wasi-nn
//     let graph = match wasi_nn::GraphBuilder::new(
//         wasi_nn::GraphEncoding::Ggml,
//         wasi_nn::ExecutionTarget::AUTO,
//     )
//     .config("metadata".to_string())
//     .build_from_cache("model_name".as_ref())
//     {
//         Ok(graph) => graph,
//         Err(e) => {
//             return Err(format!(
//                 "Fail to load model into wasi-nn: {msg}",
//                 msg = e.to_string()
//             ))
//         }
//     };

//     // initialize the execution context
//     let mut context = match graph.init_execution_context() {
//         Ok(context) => context,
//         Err(e) => {
//             return Err(format!(
//                 "Fail to create wasi-nn execution context: {msg}",
//                 msg = e.to_string()
//             ))
//         }
//     };

//      context.compute_single()

// }

use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {

    let start = std::time::Instant::now();
    let option = InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        ..Default::default()
    };
    let model: FlagEmbedding = FlagEmbedding::try_new(option)?;
    let documents = vec![
        "passage: Hello, World!",
        "query: Hello, World!",
        "passage: This is an example passage.",
        // You can leave out the prefix but it's recommended
        "fastembed-rs is licensed under MIT",
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(documents, None)?;
    let elapsed = start.elapsed().as_millis();
    print!("Elapsed: {}ms", elapsed); // -> Elapsed: 120ms

    println!("Embeddings length: {}", embeddings[0].len()); // -> Embeddings length: 4

    Ok(())
}

