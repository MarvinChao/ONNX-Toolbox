# ONNX Model Analysis
<div style="max-width: 1200px; margin: 0 auto;">

For all the supported ONNX operators, I have listed how they are broken down to each compute primitives. Also the compute cost for floating point and fixed point arithmetic will be different, particularly for activation functions. (For fixed point LUT is a common sense implementation for many nonlinear functions)

## Floating Point Compute

<div align="center">

| ONNX Operator         | MAC | ALU | EXP/LOG | DIV | TRIG | SQRT |
|:---------------------:|:---:|:---:|:-------:|:---:|:----:|:----:|
| [**Add**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add)                                     |  | $2 \times {Input\ Elements}$ |  |  |  |  |
| [**AveragePool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)                     |  | ${Output\ Elements} \times ({Kernel\ Size} - 1)$ |  | ${Output\ Elements}$ |  |  |
| [**BatchNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization)       | ${Input\ Elements}$ | $6 \times {Input\ Elements}$ |  | ${Batch\ Size} + 2 \times {Input\ Elements}$ |  | ${Batch\ Size}$ |
| [**Clip**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip)                                   |  | ${Input\ Elements} \times {Number\ of\ Bounds}$ |  |  |  |  |
| [**Concat**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat)                               |  |  |  |  |  |  |
| [**Conv**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)                                   | ${Output\ Elements} \times {Kernel Size}^2 \times \frac {Input\ Channels}{Group}$ | ${Output\ Elements}$ |  |  |  |  |
| [**Div**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div)                                     |  |  |  | ${Output\ Elements}$ |  |  |
| [**Exp**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp)                                     |  |  | ${Input\ Elements}$ |  |  |  |
| [**Gather**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)                               |  |  |  |  |  |  |
| [**GlobalAveragePool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool)         |  | ${Input\ Elements}$ |  | ${Output\ Elements}$ |  |  |
| [**Gemm**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm)                                   | ${Depends\ on\ tansA\ and\ transB}$ | ${Output\ Elements}$ |  |  |  |  |
| [**InstanceNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization) | ${Input\ Elements}$ | $6 \times {Input\ Elements}$ |  | $2 \times {B_{input} \times C_{input}} + {Input\ Elements}$ |  |  |
| [**LayerNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization)       | ${Input\ Elements}$ | $6 \times {Input\ Elements}$ |  | $2 \times {B_{input}} + {Input\ Elements}$ |  |  |
| [**LeakyRelu**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu)                         |  | $2.5 \times {Input\ Elements}$ |  |  |  |  |
| [**Log**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log)                                     |  |  | ${Input\ Elements}$ |  |  |  |
| [**Maxpool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Maxpool)                             |  | ${Output\ Elements}$ | |  |  |  |
| [**Mish**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mish)                                   | ${Input\ Elements}$ |  | $2 \times {Input\ Elements}$ | ${Input\ Elements}$ | ${Input\ Elements}$ |  |
| [**Mul**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul)                                     |  | ${Input\ Elements}$ |  |  |  |  |
| [**Pad**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad)                                     |  | ${Output\ Elements}$ |  |  |  |  |
| [**Relu**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu)                                   |  |   $0.5 \times  {Input\ Elements}$  |  |  |  |  |
| [**Reshape**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape)                             |  |  |  |  |  |  |
| [**Resize**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize)                               |  | ${Depends\ on\ resize\ mode}$ |  |  |  |  |
| [**Sigmoid**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid)                             |  | ${Input\ Elements}$ | ${Input\ Elements}$ | ${Input\ Elements}$ |  |  |
| [**Slice**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice)                                 |  |  |  |  |  |  |
| [**Softmax**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax)                             |  | ${Depends\ on\ axis}$ | ${Input\ Elements}$ | ${Depends\ on\ axis}$ |      |      |
| [**Softplus**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus)                           |  | ${Input\ Elements}$ | $2 \times {Input\ Elements}$ |     |      |      |
| [**Sqrt**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt)                                   |  |  |  |  |  | ${Input\ Elements}$ |
| [**Sub**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub)                                     |  | ${Input\ Elements}$ |  |  |  |  |
| [**Tanh**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh)                                   |  |  |  |  | ${Input\ Elements}$ |  |
| [**Transpose**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose)                         |  |  |  |  |  |  |

</div>

## Fixed Point Compute

<div align="center">

| ONNX Operator         | MAC | ALU | EXP/LOG | DIV | TRIG | SQRT |
|:---------------------:|:---:|:---:|:-------:|:---:|:----:|:----:|
| [**Add**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add)                                     |  |  |  |  |  |  |
| [**AveragePool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)                     |  |  |  |  |  |  |
| [**BatchNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization)       |  |  |  |  |  |  |
| [**Clip**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip)                                   |  |  |  |  |  |  |
| [**Concat**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat)                               |  |  |  |  |  |  |
| [**Conv**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)                                   |  |  |  |  |  |  |
| [**Div**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div)                                     |  |  |  |  |  |  |
| [**Exp**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp)                                     |  |  |  |  |  |  |
| [**Gather**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)                               |  |  |  |  |  |  |
| [**GlobalAveragePool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool)         |  |  |  |  |  |  |
| [**Gemm**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm)                                   |  |  |  |  |  |  |
| [**InstanceNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization) |  |  |  |  |  |  |
| [**LayerNormalization**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization)       |  |  |  |  |  |  |
| [**LeakyRelu**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu)                         |  |  |  |  |  |  |
| [**Log**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log)                                     |  |  |  |  |  |  |
| [**Maxpool**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Maxpool)                             |  |  |  |  |  |  |
| [**Mish**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mish)                                   |  |  |  |  |  |  |
| [**Mul**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul)                                     |  |  |  |  |  |  |
| [**Pad**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad)                                     |  |  |  |  |  |  |
| [**Relu**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu)                                   |  |  |  |  |  |  |
| [**Reshape**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape)                             |  |  |  |  |  |  |
| [**Resize**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize)                               |  |  |  |  |  |  |
| [**Sigmoid**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid)                             |  |  |  |  |  |  |
| [**Slice**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice)                                 |  |  |  |  |  |  |
| [**Softmax**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax)                             |  |  |  |  |  |  |
| [**Softplus**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus)                           |  |  |  |  |  |  |
| [**Sqrt**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt)                                   |  |  |  |  |  |  |
| [**Sub**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub)                                     |  |  |  |  |  |  |
| [**Tanh**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh)                                   |  |  |  |  |  |  |
| [**Transpose**](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose)                         |  |  |  |  |  |  |

</div>

</div>
