CIMmetry-1616 is a custom-designed IP, highly optimized edge hardware accelerator bridging the gap between high-level Machine Learning architectures and bare-metal silicon software and RTL design. Designed specifically for low-latency Edge AI, this project implements a hardware-software co-design paradigm to execute edge Convolutional Neural Network (CNN) inference as well as primitive transformers architecture operations natively on an Altera DE2-115 FPGA.

By utilizing a custom Compute-In-Memory and Systolic Array architecture, CIMmetry-1616 drastically reduces memory bottlenecks, optimizing the Multiply-Accumulate (MAC) operations required for tasks like real-time facial recognition. A special vector processing unit PRISM-16 aids the system to perform parallel activation function operation and element-wise matrix multiplication to deliver the most profound and optimum result in edge AI inference.

The project also demonstrated edge ML model training via quantization aware training(int8) in PyTorch and method in solving data scarcity problem via modern image generation model (Rectified Flow Matching Diffusion Transformer DiT) 4 data per class -> 50 data per class.

---

##System Architecture

The project is split into two primary domains: the **RTL Hardware Logic** and the **Software Control Bridge**.

### 1. Hardware (The Silicon Engine)
* **CIMmetry-1616 Core (16x16 Systolic MAC Array):** The core custom spatial matrix IP computational engine. Written entirely in Verilog HDL, this array processes 256 parallel matrix multiplications simultaneously with extreme efficiency, passing weights and activations through the grid.
* **PRISM-16 (Parallel ReLU Integer & SwiGLU Macroblock)** Another custom vector processing core IP capable of executing parallel activation function operation( ReLU and Swish) as well as 16 element-wise multiplication to cater regular ReLU required architecture and more specifically SwiGLU-type transformer architecture.
* **Compute-in-Memory (CIM) Principles:** Minimizes the energy-expensive data movement between the processor and external RAM by localizing the weight buffering directly adjacent to the MAC units. 
* **Avalon-MM / Memory Mapped Interconnects:** Handles the high-bandwidth data pipelining required to feed the systolic array without starving the compute units.

### 2. Software (The C-Bridge)
* **Low-Level Memory Addressing:** Custom Embedded C drivers that control the hardware routing and memory bandwidth management.
* **Inference Pipeline:** The host software processes the raw image tensors, flattens them, and dispatches the memory-mapped instructions to the FPGA fabric for accelerated computation.
* **PyTorch-like Usage:** The C code provides custom data structure for the matrices to be multiplied and the product of multiplication as well as a wrapper function that only takes in two tensor matrices as arguments so that users can use it easily without the hassle of memory map addressing.

### 3. The Models 
* **Alex vs Steve Siamese Network (CNN):** A custom 128 parameters TinyML edge facial-recognition model. Work to classify 8x8 image of Minecraft Steve and Alex as well as noise. Quantization-Aware-Trained (int8) via contrastive loss and 50 data for the 2 face classes.
* **Olivetti Faces Siamese Network (CNN_2):** A custom 384 parameters TinyML edge facia-recognition model. Trained with 40 different faces from Olivetti Face Dataset from sklearn library image size of 32x32. Ability to identify the identity of different individuals. QAT trained.

### 4. Miscellaneous
* **Rectified Flow-Matching for Alex vs Steve Siamese Network (Diffusion Transformer DiT):** Due to scarce data obtained (4 per classes), 2 separate flow-matching model are engineered to generate 46 more varied data for the 2 face classes respectively. The 4 images in the Data folder are all generated via the DiT.
