const sharp = require("sharp");
const ort = require("onnxruntime-node");

/**
 * Preprocesses an image buffer and returns an ONNX Tensor suitable for model input.
 * @param {Buffer} imageBuffer - The raw image buffer.
 * @returns {Promise<ort.Tensor>} - The preprocessed image tensor.
 */
async function preprocessImage(imageBuffer) {
  // Resize the image and remove the alpha channel if present
  const { data, info } = await sharp(imageBuffer)
    .resize(224, 224) // Adjust the size as per your model's requirements
    .removeAlpha() // Removes alpha channel if the image has one
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;

  // Normalize pixel values to [0, 1]
  let floatArray = Float32Array.from(data).map((v) => v / 255);
  // console.log(floatArray);

  // Apply mean and standard deviation normalization
  floatArray = normalize(floatArray);

  tensorArray = new ort.Tensor("float32", floatArray, [1, channels, height, width]);

  // Create a tensor in NCHW (batch, channels, height, width) format
  return tensorArray;
}

/**
 * Normalizes image data using mean and standard deviation.
 * @param {Float32Array} data - The image data as a flat array.
 * @returns {Float32Array} - The normalized image data.
 */
function normalize(data) {
  const mean = [0.485, 0.456, 0.406]; // Adjust these values based on your model
  const std = [0.229, 0.224, 0.225]; // Adjust these values based on your model
  const normalizedData = new Float32Array(data.length);

  for (let i = 0; i < data.length; i += 3) {
    // Normalize each channel
    normalizedData[i] = (data[i] - mean[0]) / std[0]; // Red channel
    normalizedData[i + 1] = (data[i + 1] - mean[1]) / std[1]; // Green channel
    normalizedData[i + 2] = (data[i + 2] - mean[2]) / std[2]; // Blue channel
  }
  // console.log("normalized", normalizedData);

  return normalizedData;
}

module.exports = { preprocessImage };
