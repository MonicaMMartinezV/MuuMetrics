import torch
import timm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your .pth model
pth_path = r"Evaluation\website\final_model.pth"

# Load checkpoint
ckpt = torch.load(pth_path, map_location=device)

# Rebuild model
model = timm.create_model("convnext_base", pretrained=False, num_classes=1).to(device)
state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
model.load_state_dict(ckpt[state_dict_key])
model.eval()

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 384, 384).to(device)

# Create ONNX path in the same folder as the .pth file
onnx_path = os.path.join(os.path.dirname(pth_path), "model.onnx")

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX model exported successfully to: {onnx_path}")
