# Neural ODE to Multimodal FastMRI
*Based on: Zhang, Han, Xiong, et al. (2024)*

## Introduction:

Generating high resolution frequency domain images based on accelerated $k$-space images is desirable in order for quicker diagnosis in emergent patients and reduction of motion artifacts for Pediatric cases. 
Classical deep learning approaches for solving this problem have included **U-Nets**, **GANs** and **Variational Autoencoders(VAES)**. However, GANs are known to suffer from poor generalizability and unstable training, requiring a lot of manual tuning. Meanwhile VAEs need to calculate computational expensive Jacobian matrices, lowering their tractability.

More recently *denoising diffusion probabilistic models* (or simply *diffusion* models) have been introduced. They consist of two stage, a forward diffusion stage (where the information in the image is progressively destroyed by adding Gaussian noise) and the reverse denoising stage (where the information is recovered through denoising, leveraging Convolutional neural Networks.) While diffusion models have been applied to the Fast MRI problem, they are known to suffer from sensitivity to initial conditions and they lack the ability to maintain the structural information. Furthermore, they are generally computationally intractable and multi-modality will likely consume even more compute. 

The authors make a ODE based flow-optimization model, which has the following properties:
- Follows a straight path (Conditional Optimal Transport)
    - That is , $t\mathbf{x}_{0} + (1-t)\mathbf{x}_{1}$
- Faster in inference time (ODEs, not SDEs)
- Superior SSIM compared ot other Deepl Learning Methods

## Methods:

### Classical Approaches:

The deep learning approach is to find an optimal reconstruction function that inverse-maps from an acceleration masked $k$-space to un-masked high resolution $k$-space. Multiple models have been proposed:
- Automated Transform By Manifold Approximation (AUTOMAP)
- Auto-Calibrated Recurrent Neural networks (LORAKI)
- Complex Valued Kernels for Image Reconstruction (Deep Complex MRI)
and others, including one-step GANs and diffusion.

### Overview of The Neural ODE Approach

Recitified Flow demonstrated that probability flow can be viewed as the shortest straight path between two images. The authors use the method of rectified flow to recover high resolution images from acceleration masked $k$-space images. 

Let $\mathbf{y}$ represent the complex-valued, high resolution, fully sampled $k$-space acquired from the MRI image scanner. The fully sampled imaged can be achieved by 
$$\mathbf{x_{1}} = \mathcal{F}^{-1}(\mathbf{y})$$
where $\mathcal{F}^{-1}$ isthe inverse Fourier transform in $2$ or $3$ dimensions. This is the classical method MRI images are recovered from non-accelerated MRI. 

Meanwhile, let $\mathbf{M}_{\lambda}$ is a binary random mask operator with mean $\dfrac{1}{\lambda}$, where $\lambda$ corresponds to $\lambda \times$ accelerated images. Then 
$$\mathbf{x_{0}} = \mathcal{F}^{-1}(\mathbf{M}_{\lambda}\mathbf{y})$$
where $x_0$ is the accelerated MRI image (also called the 'zero-filled' image or the 'undersampled' image). 

In this paper $\mathcal{M}_{4}$ and $\mathcal{M}_{8}$ are considered. The task is to find a reconstruction function that creates a flow between $\mathbf{x}_0$ and $\mathbf{x}_1$.

### Straight Flow Matching

Let $\mathbf{R}^{d}$ be the data space with data points $\mathbf{x_0}$ and $\mathbf{x_1}$. We take that $\mathbf{x_0} \sim \mathbf{p}_{0}$ and $\mathbf{x_1} \sim \mathbf{p_1}$. The flow $t \ in [0,1]$ where $\mathbf{x}(;0) = \mathbf{x}_0$ and $\mathbf{x}(;1) = \mathbf{x}_1$ is considered. Conditional Normalizing Flows are considered where
$$ \dfrac{\mathbf{d}}{\mathbf{d}t}\mathbf{x}(;t) = \mathbf{u}(;t)$$
where $\mathbf{u}$ is a time-dependent diffemorphism from $p_0$ to $p_1$. The aim of the loss function is to minimize the sum of sqaures given by
$$\min_{\mathbf{u}} \int_{0}^{1} \mathbb{E} \left[\left\| (\mathbf{u}(;t) - \mathbf{u}_{\theta}(;t)) \right\|^2 \right]$$
where $\theta$ is the learnable parameter of the flow. Unfortunately, only data from $\mathbf{p}_0$ can be accessed initially and calculation of $\mathbf{x}_1$ requires computationally intractable time integration which slows down training. 

Conditional normalizing flows simplify this processing by taking $x_t = t x_0 + (1-t) x_1$, which lowers the cost of generating $x_t$ significantly by assuming a constant speed of flow. 

### Architecture of Flow Matching

A similar structure leveraging U-nets are used, where an encoder for feature extraction and a decoder for reconstructing original image is used. 

An identical decoder to this simple model is used for the multi-modal case. The additional step in the multi-modal approach is a fusion layer to fuse the different frequency features. 

#### The Encoder Layer
A noise conditional score network (NSCN++) is used for the encoder network to extract and embed the image. 