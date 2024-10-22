# Code for TransFusion

## Abstract
CircRNA acts as a miRNA sponge, playing a crucial regulatory role in the occurrence and development of diseases. Existing computational prediction models have not sufficiently considered the intrinsic properties of molecules, along with local and global structural information. To address the above limitations, this paper proposes a Transformer-based graph fusion model TransFusion for predicting CMIs, which embeds the intrinsic semantic features of molecules into the network, integrates local and global structural information, and fuses multi-source feature. Multiple comparative experiments demonstrate that TransFusion can effectively predict potential relationships between circRNA and miRNA. Furthermore, we visualized the process of TransFusion in representing molecules, exploring the interpretability of the model in capturing and distinguishing molecular features. Case studies indicate that TransFusion, as a reliable auxiliary prediction tool, can deeply reveal the complex relationships between CMIs and diseases.
## Framework
![Alt Text](TransFusion_workflow.png)
## Hardware requirements
Training the TransFusion model does not strictly require a GPU, but having one is highly desirable for efficient performance. Therefore, proper installation of GPU drivers, including CUDA integration, is recommended.
## Setup Environment
We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
## Depedencies:
python>=3.9
numpy>=1.26.4  
pandas>=2.0.1  
transformers>=4.36.2  
torch>=2.0.1+cu117  
lightgbm>=3.3.5  
scipy>=1.10.0  
tqdm>=4.65.0  
matplotlib>=3.7.1  
## Usage:
### Model Usage Steps

1. **Generation of Negative Samples**: Generate all available negative samples.  
   *Execution Script*: `generate_negative_samples.py`

2. **Random Selection of Negative Samples**: Randomly select a number of negative samples equivalent to that of the positive samples.  
   *Execution Script*: `select_negative_samples.py`

3. **Extraction of Semantic Information**: Employ the SeqSemExtractor to derive semantic representations from the sequences of both positive and negative samples.  
   *Execution Script*: `SeqSemExtractor.py`

4. **Integration of Features**: Utilize GraphFusionNet to integrate molecular properties, local features, and global information.  
   *Execution Script*: `GraphFusionNet.py`

5. **Matching of Representations**: Align the comprehensive representations of both positive and negative samples.  
   *Execution Script*: `match_representations.py`

6. **Validation of Model Performance**: Conduct five-fold cross-validation experiments utilizing LightGBM to rigorously evaluate model performance.  
   *Execution Script*: `cross_validation.py`
## Dislaimer
This code was developed for research purposes only. The authors make no warranties, express or implied, regarding its suitability for any particular purpose or its performance.
