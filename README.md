# ONNX-Toolbox
Welcome to the ONNX-Toolbox! This repository is intend to be the swiss knief for ONNX model processing. My goal is to provide a collection of generic tools and utilities designed to simplify the processing and analysis of ONNX model files in a SW/HW agnostic way. Whether you're working on inspecting model structures, optimizing performance, or debugging issues, I hope this toolbox can provide you some practical solutions for your daily routine.

<br>
<br>

## ONNX Model Analyzer
This is the first tool that is currently being developed. The tool will provide you some basic graph level analysis of the compute and data transfer cost of the model. It has an implementation of using local data memory to reduce the system memory bandwidth. You can check the tool usage:

```
> python model_analyzer.py --help
usage: model_analyzer.py [-h] --input INPUT [--memory MEMORY] [--report]
                         [--save]

Toolbox for analyzing the ONNX model

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input ONNX model filename
  --memory MEMORY, -m MEMORY
                        Local memory size (in KBytes)
  --report, -r          Generate ONNX analysis report
  --save, -s            Saved processed onnx model

```

The compute cost in the current implementation is structured differently than most of other profiler, which either only track MAC (Multiply-ACumulate) or MAC + ALU OP. While this is a good method to characterize the model compute effort, I found that often on different kind of processors (CPU, GPU, DSP, NPU) they often come with special unit to accelerate a variety of compute primitives (i.e. exp or tanh) hance it will behave very differently on different processors. One example is the Special Function Unit (SFU) in Nvidia's Streaming Multiproceesor (SM). So I want to work on characterize the ONNX models in these important compute primitives instead of simplifying them into MAC and ALU operations. For now I am tracking the operations in the following compute primitives:

<div align="center">
  
| Compute Primitive | Remark                            |
|-------------------|-----------------------------------|
| MAC               | Multiply-Accumute Operations      |
| ALU               | Arithmetic Logic Unit Operations  |
| EXP/LOG           | Log() and Exp() Operations        |
| DIV               | Division Operations               |
| TRI               | Trigonometry Operations           |

</div>

For model data-transfer, in many of the modern hardware you will find local cache/memory to reduce the system memory bandwidth, using per-layer input/weight/output as indication of ONNX model data traffic requirement is off the reality. So I add an option to specify certain amount of local/dedicate memory for inference. What this mechanism do is to identify which ops are "**chainable**", which means it can be executed in local memory in tiles without the need to transfer all the output data out to system memory. It is a common and bare minimal optimization for inference that most HW will practice so I added to the tool. Note that I didn't meant to implement the most aggressive memory management scheme in this tool given many of them are HW/SW implementation specific.

<br>
<br>

## ONNX Model Optimizer
This tool is a unified ONNX optimizer using open-source tools including [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), [onnxoptimizer](https://github.com/onnx/optimizer), [onnxruntime](https://github.com/microsoft/onnxruntime) as well as my custom implementation for some missing optimizations such as Mish fusion. You can use the tool in the following usage:

```
❯ python graph_optimizer.py --help
usage: graph_optimizer.py [-h] --input INPUT --method METHOD [--level LEVEL] [--export]

ONNX Graph Optimization Tool

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input ONNX model filename
  --method METHOD, -m METHOD
                        Choose optimizer to run [onnxsim|onnx|ort|onnx-toolbox|all]
  --level LEVEL, -l LEVEL
                        Specify optimization level [basic|extend|all]
  --export, -e          Export optimized ONNX model
```

If you select the method to be **all**, this is the sequence of the model optimizations the tool will perform:
* **onnx-simplifier**
* **onnxoptimizer**

  I only made a conservative selection of the supported passes
  
* **onnxruntime**

  The default level is all, which includes basic + extended + layout
  
* **ONNX-Toolbox**

  Only fuse-mish is applied right now

<br>
<br>

Using these tools is always a little bit tricky. I found that each of them will break at particular graphs so they are not always reliable. My recommendation is take trial-and-error approach. 

In addition, I tried to verify the graph as much as I could and use ONNX Runtime to perform an inference test but the only way to proof the correctness is to perform accuracy test, which is hard and resource intensive. For production use the additional accuracy test will be highly recommended.
