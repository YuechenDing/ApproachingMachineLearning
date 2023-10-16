"""
CNN:
    - conv; initialization + activation function
    - build models by inherit nn.Module, overwrite __init__, forward
    - build image dataset by overwrite __init__, __len__, __getitem__
    - segmentation_models_pytorch,  wtfml.engine.Engine
    - code architecture: 
        - dataset.py (customized dataset)
        - engine.py (train/evaluation)
        - model.py (customized models/model dict)
        - train.py (main)

"""