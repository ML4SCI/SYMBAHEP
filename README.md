# SYMBA-FI

Repository for SYMBA-FI: An offshoot of the main SYMBA project with the goal of extending SYMBA's capabilities to calculate square amplitudes directly from potentially Feynman Diagrams/Feynman Integrals (FIs). This project was started as part of Google Summer of Code (GsoC) 2023 as part of the Machine Learning for Science (ML4SCI) umbrella organization. Work on SYMBA-FI will continue beyond the GsoC 2023 period.
This commit is the final commit for the GsoC 2023 coding period.

**GOAL:**

The specific goal of SYMBA-FI is to modify the SYMBA sequence-to-sequence transformer model to make use of symbolic matrix input.
In the process of numerically solving FIs, matrices representing linear systems of equations and/or ordinary differential equations are set up and solved. Naturally, these matrices contain information about their unique associated FIs. motivatied by recent investigation into using seq-2-seq transformers for symbolic linear algebraic problems [Charton, 2021](https://arxiv.org/abs/2112.01898), we aim to modify SYMBA to accept symbolic matrices representative of given FIs or diagrams as input to the model and train it to return the correct symbolic squared amplitudes.

**WHAT'S BEEN DONE:**

A large amount of time was dedicated to learning the mathematical theory behind systematically solving Feynman Integrals. The initial estimate was that this would be complete by July, but more time was needed to fully understand where in the process the linear algebraic information could be found. The main research part was not finished until August, and some additional investigation is ocasionally required for specific points as needed.
To create a functional SYMBA-FI, a dataset of input data matrices and labeled target squared amplitudes must be created. To this end we are utilizing [AMFlow](https://gitlab.com/multiloop-pku/amflow), a mathematica package for solving FIs, and planning to modify the code to extract the matrix information for the FIs.
Currently, SYMBA data is generated using [Modern ARtificial Theoretical phYsicist (MARTY)](https://github.com/docbrown1955/marty-public). Written in C++, MARTY is capable of complex physics calculations, mainly amplitude and squared amplitude calculations. The current work in progress is modifying existing code to generate AMFlow input files for the Feynman diagrams that MARTY generates. Currently, a semi-functional code for data generation has been written. Details can be found below in the "MERGED CODE" section. 

**WHAT NEEDS TO BE DONE:**

Dataset generation is still currently in development. SYMBA uses MARTY to generate datasets of amp/sqamp and diagram/sqamp. However, AMFlow needs FI information in a very specific form, slightly different from the standard form in which physicists are used to writing FIs. It's also not so straightforward to extract the needed information from MARTY by default. Currently, the focus is on exploring MARTY more in depth to identify how to extract said information or explicitly construct FIs in a form more suitable for AMFlow. Specifically, we need to extract propagators from FIs in the form that AMFlow expects them. We will also generalize dataset generation from 2-to-2 processes to general n-to-m.
Some recent searching shows that MARTY can explicitly construct propagators so that they can be printed as simple expressions or string output. This likely happens before the amplitude objects are created, meaning that more extensive modification or re-writing of the data generation code is necessary to obtain the necessary propagator information in this way. This may also make it possible to fully contain the dataset generation to MARTY alone. This is part of the current avenue of investigation.

If appropriately formatted AMFlow input can be generated, we must then modify AMFlow to extract the desired matrix information for the FIs in the dataset.
An alternative approach would be to use another software package/suite such as FeynCalc to extract this propagator information, as some preliminary research suggests that FeynCalc has built-in functions to directly return propagators in the desired format. As FeynCalc is written in mathematica, it may be easier to interface with AMFlow, but this may have an adverse effect on the speed and efficiency of data generation at scale.

Once datasets have been created, SYMBA-FI must be created and trained. SYMBA-FI is to be built and trained using Neeraj Anand's [PyTorch implementation of SYMBA](https://github.com/ML4SCI/SYMBA_Pytorch) as a basis. In accordance with existing version(s) of SYMBA, SYMBA-FI will be evaluated on the basis of metrics including sequence accuracy and token accuracy. 
After building/training SYMBA-FI, we may also consider avenues currently being investigated for the main version of SYMBA, including modifications and improvements for accuracy predicting longer sequences and alternative attention mechanisms.

**MERGED CODE:**

The current repository consists of a combination of old and newer code. The older code are demos from older versions of SYMBA.
The recently merged code is a work in progress of the data generation code. It can be found in the folder "marty_data_generation/QED/," and consists of "All_ParticlesIO.cpp" and "QED_loop_insertions.py", the C++ code that uses MARTY for generating data and the python script that specifies which particles and processes to consider, respectively. This code is based off of Marco Knipfer's data generation code for [SYMBA-prefix](https://github.com/ML4SCI/SYMBA/tree/main) and has been modified where appropriate for SYMBA-FI. While Marco's code is also outdated, it is similar enough to current data generation code for SYMBA that no problems should arise.
This code uses MARTY to create a single large .wl (WolframScript) file with the general structure for running AMFlow input for all given integrals. Eventually, it should produce a single .wl file for each individual FI to be used in the dataset. Of the major information needed by AMFlow, currently the internal/external momenta are extracted from diagrams and provided by MARTY. Obtaining propagator information is the current focus. Conservation and replacement rules are hard-coded placeholder for 2-to-2 processes. Numerical information in the outputs are also placeholder, since we will not require it for SYMBA-FI's goals. Processes have been heavily restricted to electron scattering for code testing and development purposes. 

**LESSONS LEARNED:**

Although I should have expected it, I have learned that project complexity and scope can suddenly increase unexpectedly. Despite learning the needed theory for solving FIs and ML, there is still a gap to bridge between MARTY and AMFlow that was unexpectedly large, requiring that I learn both C++ and the more complex functions of MARTY just to begin setting up the needed dataset. I also learned that I should be more aware of version control when utilizing software that depends on multiple auxiliary programs, as a version control issue set me back 3 weeks before I could identify the issue properly.
