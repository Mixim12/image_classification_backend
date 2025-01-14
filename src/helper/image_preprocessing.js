const sharp = require("sharp");
const ort = require("onnxruntime-node");

async function preprocessImage(imageBuffer) {
  const { data, info } = await sharp(imageBuffer).resize(224, 224).removeAlpha().raw().toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;

  const floatData = Float32Array.from(data).map((v) => v / 255);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const normalizedData = new Float32Array(channels * height * width);

  for (let c = 0; c < channels; c++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const idx = h * width * channels + w * channels + c;

        normalizedData[c * height * width + h * width + w] = (floatData[idx] - mean[c]) / std[c];
      }
    }
  }

  const inputTensor = new ort.Tensor("float32", normalizedData, [1, channels, height, width]);

  return inputTensor;
}

module.exports = { preprocessImage };
