const sharp = require('sharp');
const ort = require('onnxruntime-node');
let session;
const path = require('path');
const pathOnnx = path.join(__dirname, '..', '..', 'model.onnx');

async function loadModel() {
  session = await ort.InferenceSession.create(pathOnnx);
}

// Convert image buffer to tensor
async function imageToTensor(imagePath) {
  // Resize and get raw RGB
  const { data, info } = await sharp(imagePath)
    .resize(384, 384)
    .raw()
    .toBuffer({ resolveWithObject: true });

  // data is [H * W * C], we need [1, C, H, W]
  const floatData = new Float32Array(3 * info.height * info.width);

  for (let i = 0; i < info.height * info.width; i++) {
    floatData[i] = data[i * 3] / 255;          // R
    floatData[i + info.height * info.width] = data[i * 3 + 1] / 255; // G
    floatData[i + 2 * info.height * info.width] = data[i * 3 + 2] / 255; // B
  }

  return new ort.Tensor('float32', floatData, [1, 3, info.height, info.width]);
}

async function discretizeValue(bcsPred) {
  // Clamp between 1.0 and 5.0
  if (bcsPred > 5.0) bcsPred = 5.0;
  if (bcsPred < 1.0) bcsPred = 1.0;

  // Round to nearest 0.25
  return Math.round(bcsPred * 4) / 4;
}
// Make prediction
async function predict(imagePath) {
  const tensor = await imageToTensor(imagePath);
  const feeds = { input: tensor }; // must match input name
  const results = await session.run(feeds);
  
  // results.output.data is an array of predictions
  const rawPred = results.output.data[0]; // get the first value if batch size 1
  const discretizedPred = discretizeValue(rawPred);
  
  return discretizedPred;
}


// Example
const imagePath = path.join(__dirname, 'downloads', 'images', '1111_test.jpg');
(async () => {
  await loadModel();
  const prediction = await predict(imagePath);
  console.log("Predicted BCS:", prediction);
})();

module.exports = { predict };