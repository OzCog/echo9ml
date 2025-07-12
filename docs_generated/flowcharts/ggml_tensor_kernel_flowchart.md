# ggml_tensor_kernel Module Flowchart

```mermaid
graph TD
    ggml_tensor_kernel[ggml_tensor_kernel]
    ggml_tensor_kernel_TensorOperationType[TensorOperationType]
    ggml_tensor_kernel --> ggml_tensor_kernel_TensorOperationType
    ggml_tensor_kernel_TensorMetadata[TensorMetadata]
    ggml_tensor_kernel --> ggml_tensor_kernel_TensorMetadata
    ggml_tensor_kernel_CognitiveTensor[CognitiveTensor]
    ggml_tensor_kernel --> ggml_tensor_kernel_CognitiveTensor
    ggml_tensor_kernel_CognitiveTensor___post_init__[__post_init__()]
    ggml_tensor_kernel_CognitiveTensor --> ggml_tensor_kernel_CognitiveTensor___post_init__
    ggml_tensor_kernel_CognitiveTensor_to_dict[to_dict()]
    ggml_tensor_kernel_CognitiveTensor --> ggml_tensor_kernel_CognitiveTensor_to_dict
    ggml_tensor_kernel_CognitiveTensor_from_dict[from_dict()]
    ggml_tensor_kernel_CognitiveTensor --> ggml_tensor_kernel_CognitiveTensor_from_dict
    ggml_tensor_kernel_GGMLTensorKernel[GGMLTensorKernel]
    ggml_tensor_kernel --> ggml_tensor_kernel_GGMLTensorKernel
    ggml_tensor_kernel_GGMLTensorKernel___init__[__init__()]
    ggml_tensor_kernel_GGMLTensorKernel --> ggml_tensor_kernel_GGMLTensorKernel___init__
    ggml_tensor_kernel_GGMLTensorKernel__document_semantic_mappings[_document_semantic_mappings()]
    ggml_tensor_kernel_GGMLTensorKernel --> ggml_tensor_kernel_GGMLTensorKernel__document_semantic_mappings
    ggml_tensor_kernel_GGMLTensorKernel__initialize_tensor_shapes[_initialize_tensor_shapes()]
    ggml_tensor_kernel_GGMLTensorKernel --> ggml_tensor_kernel_GGMLTensorKernel__initialize_tensor_shapes
    ggml_tensor_kernel_GGMLTensorKernel__register_custom_operations[_register_custom_operations()]
    ggml_tensor_kernel_GGMLTensorKernel --> ggml_tensor_kernel_GGMLTensorKernel__register_custom_operations
    ggml_tensor_kernel_GGMLTensorKernel_create_tensor[create_tensor()]
    ggml_tensor_kernel_GGMLTensorKernel --> ggml_tensor_kernel_GGMLTensorKernel_create_tensor
    style ggml_tensor_kernel fill:#99ff99
```