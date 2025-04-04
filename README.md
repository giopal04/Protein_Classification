# Progression

### Steps to do:
1. #### Define a DataSet
    
    - Study the number of points for model and class <span style="color:green">**DONE**</span>  
    - Remove the duplicate elements (?)  
    - Add Sampler to make the dataset more omogeneous <span style="color:yellow">**DONE**</span>  
    &emsp;Make it runtime in the Dataloader, refactor the code 
    &emsp;Problem:  
    &emsp;&emsp;How to deal with the fact that sampling different meshes with different number of points will results  
    &emsp;&emsp;in input pointclouds with differents density  
    &emsp;&emsp;Check if a sampler exist in <code>pyvista</code> _there is but seems to be too refined respect to what we need_
    - Normalize the data in a unit sphere <span style="color:yellow">**DONE**</span>  
    &emsp;Make it runtime in the Dataloader, refactor the code 
    - Create multiple dataset (more omogeneous) <span style="color:green">**DONE**</span>  
    - Get a function that allow to select partial dataset <span style="color:green">**DONE**</span> (function implemented)  
    - Make DataSet be able to load compressed numpy <span style="color:green">**DONE**</span>  

1. #### Implement the DataLoader
    
    - Introduce a modified DataLoader  
    This is not necessary, the augmentation are implemented in the <code>def \_\_getitem__()</code> method of the <code>ProteinDataset</code>
    - Augmentations:  <span style="color:green">**DONE**</span>
        - Traslations
        - Normalization
        - Rotations 
        - Noise
        - Sampling
    - Still need to modify <code>UnitSphereNormalization()</code> and <code>RandomSampler()</code> to not take the index variable in the <code>\_\_call__()</code> method <span style="color:yellow">**DONE**</span> (to improve)

1. #### Define a training loop

    - Assemble the model (encoder: pointnet, decoder: ?) <span style="color:green">**DONE**</span>  
    - Perform first runs with the small dataset <span style="color:green">**DONE**</span>  
    - Iplement a way to get the validation set <span style="color:green">**DONE**</span> 
    - Write down metrics <span style="color:green">**DONE**</span> 
    - Log everything in <code>wandb</code>  
    Define a dict to log everything
    - Make a function for the training loop and validation <span style="color:green">**DONE**</span>

1. #### General things to do  
    
    - Clean the code and the notebook  

### Considerations

We have found multiple disconnected meshes characterized by few vertices,  
currently we are cutting disregarding every mesh with less than 5000 points.

The dataset has a really dishomogeneous distribution of meshes per class,  
at this moment we have restricted the dataset to the subset where each class have more than a 100 meshes,  
finally we have taken a fixed number of meshes for each class.

<span style="color:orange">First run</span> done on a small dataset (_4 classes, 384 images, no validation_) with some promising results, it seemed to be learning somthing. Computing the validation loss it was clear that **nothing was learned**.

Polish everything

Finally we had been able to make the <code>PointNet</code> learn something  

<code>PointNet</code> still has troubles learning more than two classes, in particular if it has to deal with an unbalanced dataset.

### Want to try

1. Weight in loss to balance the dataset  
1. Duplicate less populated classes to balance the dataset  
1. Pass to the <code>PointNet</code> decoder the features ectracted from its encoder and a <code>ResNet</code> fine tuned on the proteins screenshot

### Good runs

- Small dataset of **194** protein and **2** classes, <code>shrec_run-GPU-2-IMPARA.ipynb</code>, reached over **0.95** of accuracy  
- Dataset containing **1000** proteins divided among **10** classes, <code>shrec_run-GPY-10_clss-1000_images.ipynb</code>, reached **0.65** of accuracy (was overfitting badly)