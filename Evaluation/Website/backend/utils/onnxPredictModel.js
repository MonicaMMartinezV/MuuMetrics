const sharp = require('sharp');
const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs');

// Path to the ONNX model file
const pathOnnx = path.join(__dirname, '..', '..', 'model.onnx');
let session;
let modelReady = false;

// =============================================================
// MODEL LOADING AND INITIALIZATION
// =============================================================

/**
 * Loads the ONNX model into an InferenceSession.
 */
async function loadModel() {
    if (modelReady) {
        console.log("ONNX Model already loaded.");
        return;
    }
    try {
        if (!fs.existsSync(pathOnnx)) {
             throw new Error(`Model file not found at: ${pathOnnx}`);
        }
        session = await ort.InferenceSession.create(pathOnnx);
        modelReady = true;
        console.log("ONNX Model loaded successfully.");
    } catch (error) {
        console.error("Error loading ONNX model:", error.message);
        throw error;
    }
}

// =============================================================
// IMAGE PREPROCESSING
// =============================================================

/**
 * Converts an image file buffer to a preprocessed ONNX tensor.
 * The model expects input shape [1, 3, 384, 384] and normalized float32 values.
 * @param {string} imagePath - The local path to the image file.
 * @returns {Promise<ort.Tensor>} The input tensor for the ONNX model.
 */
async function imageToTensor(imagePath) {
    if (!fs.existsSync(imagePath)) {
        throw new Error(`Image file not found at: ${imagePath}`);
    }
    
    // Resize to 384x384 and get raw RGB data
    const { data, info } = await sharp(imagePath)
        .resize(384, 384)
        .raw()
        .toBuffer({ resolveWithObject: true });

    // data is [H * W * C], we need [1, C, H, W] for the model
    const floatData = new Float32Array(3 * info.height * info.width);
    const pixelCount = info.height * info.width;

    // Convert and normalize (0-255 -> 0-1) for C, H, W layout
    for (let i = 0; i < pixelCount; i++) {
        // R (Channel 0)
        floatData[i] = data[i * 3] / 255.0; 
        // G (Channel 1)
        floatData[i + pixelCount] = data[i * 3 + 1] / 255.0; 
        // B (Channel 2)
        floatData[i + 2 * pixelCount] = data[i * 3 + 2] / 255.0; 
    }

    return new ort.Tensor('float32', floatData, [1, 3, info.height, info.width]);
}

// =============================================================
// POSTPROCESSING
// =============================================================

/**
 * Discretizes the raw BCS prediction to the nearest 0.25 and clamps it 
 * between 1.0 and 5.0.
 * @param {number} bcsPred - The raw prediction value from the model.
 * @returns {number} The discretized and clamped BCS value.
 */
function discretizeValue(bcsPred) {
    // 1. Clamp between 1.0 and 5.0
    if (bcsPred > 5.0) bcsPred = 5.0;
    if (bcsPred < 1.0) bcsPred = 1.0;

    // 2. Round to nearest 0.25 (by multiplying by 4, rounding, then dividing by 4)
    return Math.round(bcsPred * 4) / 4;
}

// =============================================================
// MAIN PREDICTION FUNCTION
// =============================================================

/**
 * Performs BCS prediction on a given image file path.
 * @param {string} imagePath - The local path to the image file.
 * @returns {Promise<number>} The predicted and discretized BCS value.
 */
async function predict(imagePath) {
    // 🔥 FIX: Ensure model is loaded before attempting inference
    if (!session) {
        await loadModel(); 
    }
    
    const tensor = await imageToTensor(imagePath);
    // The input name 'input' must match the model's expected input name
    const feeds = { input: tensor }; 
    
    const results = await session.run(feeds);
    
    // results.output.data is typically a Float32Array containing predictions
    // Assuming the output name is 'output' and the prediction is the first element
    const rawPred = results.output.data[0]; 
    
    const discretizedPred = discretizeValue(rawPred);
    
    console.log(`BCS Prediction (Raw: ${rawPred.toFixed(3)}, Discretized: ${discretizedPred})`);

    return discretizedPred;
}


// Export the primary function
module.exports = { predict };