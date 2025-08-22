# XNOR-SRAM: In-Memory Computing for Binary/Ternary Neural Networks

This project explores the **XNOR-SRAM architecture**, an **in-memory computing (IMC)** solution proposed for **binary and ternary neural networks (BNNs/TNNs)**. The architecture integrates **matrix–vector multiplication (MVM)** functionality directly within **SRAM bit-cells**, enabling efficient neural network inference while reducing data movement overhead.

---

## 🔬 Project Goals
- **Analyze physical design challenges** in XNOR-SRAM arrays:
  - Pull-Up / Pull-Down (PU/PD) resistance mismatch
  - Spatial gradient variations across the memory array
- **Evaluate the impact** of these variations on computational accuracy.
- **Propose correction techniques** that:
  - Improve computational integrity  
  - Require no hardware modifications  
  - Use algorithmic or architectural compensation

---

## ⚙️ Methods
1. **Modeling SRAM-based IMC**
   - Simulated MVM operations under PU/PD mismatch and spatial gradients.  
   - Analyzed voltage reference bit-line (VRBL) distributions.  

2. **Error Characterization**
   - Compared raw vs balanced inputs.  
   - Evaluated correction techniques for restoring accuracy.  

3. **Mitigation**
   - Lightweight **post-processing corrections** that reduce error without redesigning the SRAM cells.  

---
