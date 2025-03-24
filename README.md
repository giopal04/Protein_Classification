## Progression

#### Steps to do:
1. ##### Define a DataLoader
    
    - Study the number of points for model and class <span style="color:green">**DONE**</span>  
    - Remove the duplicate elements (?)  
    - Add Sampler to make the dataset more omogeneous <span style="color:green">**DONE**</span>  
    _Clean the sampler and its inputs_
    &emsp;Problem:  
    &emsp;&emsp;How to deal with the fact that sampling different meshes with different number of points will results  
    &emsp;&emsp;in input pointclouds with differents density  
    &emsp;&emsp;Check if a sampler exist in <code>pyvista</code> _there is but seems to be too refined respect to what we need_
    - Normalize the data in a unit sphere <span style="color:green">**DONE**</span>  
    _Just need to clean the normalizer and its inputs_
    - Create multiple dataset (more omogeneous) <span style="color:green">**DONE**</span>  
    - Get a function that allow to select partial dataset <span style="color:green">**DONE**</span> (function implemented)  
    - Implement in the <code>DataSet</code> the augmentations (__which ones?__)

2. ##### Define a training loop

    - Assemble the model (encoder: pointnet, decoder: ?) <span style="color:green">**DONE**</span>  
    - Perform first runs with the small dataset <span style="color:green">**DONE**</span>  
    - Iplement a way to get the validation set <span style="color:green">**DONE**</span> 
    - Write down metrics <span style="color:green">**DONE**</span> 
    - Log everything in <code>wandb</code>  
    Define a dict to log everything

#### Considerations

We have found multiple disconnected meshes characterized by few vertices,  
currently we are cutting disregarding every mesh with less than 5000 points.

The dataset has a really dishomogeneous distribution of meshes per class,  
at this moment we have restricted the dataset to the subset where each class have more than a 100 meshes,  
finally we have taken a fixed number of meshes for each class.

<span style="color:orange">First run</span> done on a small dataset (_4 classes, 384 images, no validation_) with some promising results,  
it seemed to be learning somthing 

Polish everything