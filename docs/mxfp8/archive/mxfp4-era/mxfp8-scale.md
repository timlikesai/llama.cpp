
do mxfp4 and mxfp8 use the same scale formula for selecting the block scale


AI Mode
All
Images
Videos
Forums
Shopping
Short videos
More
Tools
AI Overview
Yes, MXFP4 and MXFP8 use the exact same mathematical scale formula for selecting the block scale. 

NVIDIA Docs
Both formats are governed by the OCP Microscaling Formats (MX) Specification, meaning they share the same architectural rules for scale calculation. In both cases, a block size of 32 elements shares a single 8-bit E8M0 scale factor (which is a pure power-of-two exponent). 

Open Compute Project
 +3
The Block Scale Formula
The general mathematical formula used to calculate the block scale factor (scale) for both MXFP4 and MXFP8 is defined by: 

NVIDIA Docs

The underlying steps are identical, but the formula evaluates slightly differently depending on the specific output data type (
): 

NVIDIA Docs
Find the Block Maximum (amax): The system scans the block of 32 elements to find the absolute maximum value (amax(vals)). 

NVIDIA Docs
 +1
Normalize by the Output Target Data Type (
): The block max is divided by the maximum representable finite value of the element data type:
For MXFP4: The element type is E2M1, where 

.
For MXFP8: The element type can be E4M3 (where 

) or E5M2 (where 

). 

arXiv.org
 +5
Quantize and Round Up: The resulting value is rounded up to the nearest representable power-of-two step allowed by the E8M0 scale format. 

ZeroEntropy
 +1
Summary of Component Differences
Metric / Parameter	MXFP4	MXFP8
Scale Selection Formula	Same	Same
Block Size	32 elements	32 elements
Scale Data Type	E8M0 (8-bit power-of-two)	E8M0 (8-bit power-of-two)
Element Format	FP4 (E2M1)	FP8 (E4M3 or E5M2)

 used in formula	6.0	448 (for E4M3) or 57344 (for E5M2)
(Note: Be careful not to confuse MXFP4 with NVFP4—NVIDIA's proprietary 4-bit variant introduced with the Blackwell architecture. NVFP4 changes the block size to 16 elements and uses a more precise E4M3 FP8 scale factor rather than the power-of-two E8M0 scale used in the OCP MX standard.) 

Wikipedia
 +3
Are you trying to implement this quantization formula manually in code (such as Triton or PyTorch), or are you configuring it for hardware like NVIDIA Blackwell or AWS Neuron? I can provide the exact implementation snippet or hardware configurations if needed. 

YouTube
·Cohere
 +1

NVIDIA Docs
Block Scaling — NVIDIA cuDNN
Sep 2, 2026 — Block Scale Quantize#. The block scale quantize operation computes the quantized output and scaling factor tensors from a higher p...


GitHub Pages documentation
How Scales Are Applied in MXFP8 Attention
Apr 20, 2026 — MXFP8 vs Per-Tensor FP8 In regular FP8 (as used on Hopper), each tensor gets one scale factor shared across all elements. This mea...

Harold Benoit
NVFP4 format
Dec 6, 2025 — NVFP4 is an enhanced 4-bit format that provides improved numerical properties over MXFP4. NVFP4 has the following characteristics:

Show all

