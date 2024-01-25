#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use once_cell::sync::OnceCell;
use serde::{ Deserialize, Serialize };
use wasi_nn::{ Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType };
use std::sync::Mutex;

static MAX_BUFFER_SIZE: OnceCell<usize> = OnceCell::new();
static CTX_SIZE: OnceCell<usize> = OnceCell::new();
static GRAPH: OnceCell<Mutex<Graph>> = OnceCell::new();
static METADATA: OnceCell<Metadata> = OnceCell::new();

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Metadata {
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    #[serde(rename = "n-predict")]
    pub n_predict: u64,
    #[serde(rename = "n-gpu-layers")]
    pub n_gpu_layers: u64,
    #[serde(rename = "batch-size")]
    pub batch_size: u64,
    #[serde(rename = "temp")]
    pub temp: f32,
    #[serde(rename = "repeat-penalty")]
    pub repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    reverse_prompt: Option<String>,
}

struct Graph {
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}

impl Graph {
    pub fn new(model_alias: impl AsRef<str>, options: &Metadata) -> Self {
        let config = serde_json::to_string(&options).unwrap();

        // load the model
        let graph = wasi_nn::GraphBuilder
            ::new(wasi_nn::GraphEncoding::Pytorch, wasi_nn::ExecutionTarget::AUTO)
            .config(config)
            .build_from_cache(model_alias.as_ref())
            .unwrap();

        // initialize the execution context
        let context = graph.init_execution_context().unwrap();

        Self {
            _graph: graph,
            context,
        }
    }

    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    pub fn compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    pub fn compute_single(&mut self) -> Result<(), WasiNnError> {
        self.context.compute_single()
    }

    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T]
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }

    pub fn get_output_single<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T]
    ) -> Result<usize, WasiNnError> {
        self.context.get_output_single(index, out_buffer)
    }

    // ...

    pub async fn infer(prompt: impl AsRef<str>) -> Result<Vec<u8>, String> {
        // Use OnceCell::get to access the Graph. If it's not initialized, return an error.
        let graph = GRAPH.get().ok_or_else(|| "Graph is not initialized.".to_string())?;
        let mut graph = graph.lock().map_err(|e| e.to_string())?;

        // Encode the prompt into bytes and set it as the input tensor.
        let tensor_data = prompt.as_ref().as_bytes().to_vec();
        graph
            .set_input(0, wasi_nn::TensorType::U8, &[1, tensor_data.len()], &tensor_data)
            .map_err(|e| format!("Failed to set input tensor: {}", e))?;

        // Perform the computation (inference).
        graph.compute().map_err(|e| format!("Failed to execute model inference: {}", e))?;

        // Retrieve the output from the computation.
        let buffer_size = *MAX_BUFFER_SIZE.get().ok_or_else(||
            "MAX_BUFFER_SIZE is not set.".to_string()
        )?;
        let mut output_buffer = vec![0u8; buffer_size];
        let output_size = graph
            .get_output(0, &mut output_buffer)
            .map_err(|e| format!("Failed to get output tensor: {}", e))?;

        // Truncate the output buffer to the actual size of the output.
        output_buffer.truncate(output_size);

        Ok(output_buffer)
    }

    // ...
}
