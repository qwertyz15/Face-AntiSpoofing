import torch
from src.config import PretrainedConfig
from src.NN import MultiFTNet 

def print_model_weights(model):
    for name, param in model.state_dict().items():
        print(f"Layer: {name}")
        print(f"Values: \n{param}\n") 

def load_pytorch_model(model_path, input_size, num_classes, device_id=0):
    # Initialize the configuration
    config = PretrainedConfig(model_path, device_id=device_id, input_size=input_size, num_classes=num_classes)

    # Initialize the MultiFTNet model using the configuration
    model = MultiFTNet(img_channel=3, num_classes=config.num_classes, 
                       embedding_size=128, conv6_kernel=config.kernel_size).to(config.device)

    model = torch.nn.DataParallel(model)
    # Load the model weights
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()
    return model

if __name__ == "__main__":
    model_path = '/content/Face-AntiSpoofing/2024-01-23-11-05_AntiSpoofing_models_1.5_224_224_epoch-12.pth'  # Replace with the actual path to your model
    input_size = 224  # Replace with the input size used during training
    num_classes = 2   # Replace with the number of classes used during training
    softmax = torch.nn.Softmax(dim=1)

    model = load_pytorch_model(model_path, input_size, num_classes)  
    # print_model_weights(model)  
    # Create a random tensor of shape [1, 3, 224, 224]
    device = next(model.parameters()).device
    random_input = torch.randn(1, 3, 224, 224).to(device)

    # Get the output from the model
    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(random_input)
        print("Output:")
        print(output)
        probabilities = softmax(output)
        print(probabilities)
        # Determine the predicted class
        predicted_class = torch.argmax(probabilities, dim=1)
        print("Predicted class:", predicted_class.item())
