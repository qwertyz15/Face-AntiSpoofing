# python load_model.py --model_path ./saved_models/AntiSpoofing_bin_1.5_128.pth --device cuda:0


import torch
import argparse
from collections import OrderedDict
from src.NN import MultiFTNet, MiniFASNetV2SE

def load_model(model_path, device='cuda:0'):
    """
    Load a model from a .pth file and determine its type based on the state dictionary.

    Parameters:
    - model_path (str): Path to the .pth model file.
    - device (str): Device to load the model on. Default is 'cuda:0'.

    Returns:
    - model (torch.nn.Module): The loaded model.
    """
    # Load state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Print state dictionary keys to understand the structure
    print("State Dictionary Keys:")
    print(state_dict.keys())

    # Remove "module.model." prefix from keys if it exists
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.model.'):
            new_key = key[len('module.model.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Determine model type and parameters from new_state_dict keys
    if any('FTGenerator' in key for key in new_state_dict.keys()):
        model_type = 'MultiFTNet'
        kernel_size_key = next(key for key in new_state_dict.keys() if 'conv_6_dw.conv.weight' in key)
        kernel_size = new_state_dict[kernel_size_key].shape[2:]
    else:
        model_type = 'MiniFASNetV2SE'
        kernel_size = (5, 5)  # Default kernel size, adjust if necessary

    num_classes_key = next(key for key in new_state_dict.keys() if 'prob.weight' in key)
    num_classes = new_state_dict[num_classes_key].shape[0]

    # Initialize the appropriate model architecture
    if model_type == 'MultiFTNet':
        model = MultiFTNet(conv6_kernel=kernel_size, num_classes=num_classes).to(device)
    elif model_type == 'MiniFASNetV2SE':
        model = MiniFASNetV2SE(conv6_kernel=kernel_size, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Filter state dictionary to only include keys present in the model's state dictionary
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}

    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a Face-AntiSpoofing model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to load the model on")
    args = parser.parse_args()

    model = load_model(args.model_path, args.device)
    print("Model loaded successfully!")
