CIMmetry-1616 is a custom-designed IP, highly optimized edge hardware accelerator bridging the gap between high-level Machine Learning architectures and bare-metal silicon software and RTL design. Designed specifically for low-latency Edge AI, this project implements a hardware-software co-design paradigm to execute edge Convolutional Neural Network (CNN) inference as well as primitive transformers architecture operations natively on an Altera DE2-115 FPGA.

By utilizing a custom Compute-In-Memory and Systolic Array architecture, CIMmetry-1616 drastically reduces memory bottlenecks, optimizing the Multiply-Accumulate (MAC) operations required for tasks like real-time facial recognition.

The project also demonstrated edge ML model training via quantization aware training(int8) in PyTorch and method in solving data scarcity problem via modern image generation model (Rectified Flow Matching Diffusion Transformer DiT) 4 data per class -> 50 data per class.

---

##System Architecture

The project is split into two primary domains: the **RTL Hardware Logic** and the **Software Control Bridge**.

### 1. Hardware (The Silicon Engine)
* **16x16 Systolic MAC Array:** The core computational engine. Written entirely in Verilog HDL, this array processes 256 parallel matrix multiplications simultaneously with extreme efficiency, passing weights and activations through the grid.
* **Compute-in-Memory (CIM) Principles:** Minimizes the energy-expensive data movement between the processor and external RAM by localizing the weight buffering directly adjacent to the MAC units.
* **Avalon-MM / Memory Mapped Interconnects:** Handles the high-bandwidth data pipelining required to feed the systolic array without starving the compute units.

### 2. Software (The C-Bridge)
* **Low-Level Memory Addressing:** Custom Embedded C drivers that control the hardware routing and memory bandwidth management.
* **Inference Pipeline:** The host software processes the raw image tensors, flattens them, and dispatches the memory-mapped instructions to the FPGA fabric for accelerated computation.
* **PyTorch-like Usage:** The C code provides custom data structure for the matrices to be multiplied and the product of multiplication as well as a wrapper function that only takes in two tensor matrices as arguments so that users can use it easily without the hassle of memory map addressing.

### 3. The Models 
* **Siamese Network (CNN):** A custom 128 parameters edge facial-recognition model. Quantization-Aware-Trained (int8) via contrastive loss and 50 data for the 2 face classes.
* **Rectified Flow-Matching (Diffusion Transformer DiT):** Due to scarce data obtained (4 per classes), 2 separate flow-matching model are engineered to generate 46 more varied data for the 2 face classes respectively. The 4 images in the Data folder are all generated via the DiT.
