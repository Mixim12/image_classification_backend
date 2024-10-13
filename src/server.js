const fs = require("fs");
const ini = require("ini");
const fastify = require("fastify")({ logger: true });
const multipart = require("@fastify/multipart");
const ort = require("onnxruntime-node");
const { preprocessImage } = require("./helper/image_preprocessing");
const cors = require("@fastify/cors");
const { log } = require("console");

// Read configuration from the INI file
const config = ini.parse(fs.readFileSync("config.ini", "utf-8"));

// Parse server configuration
const serverPort = parseInt(config.server.port, 10) || 3000;
const serverHost = config.server.host || "0.0.0.0";

// Parse model configuration
const modelPath = config.model.model_path || "./model.onnx";

// Parse limits configuration
const fileSizeLimit = parseInt(config.limits.file_size, "10") || 5 * 1024 * 1024; // Default to 5 MB
console.log(fileSizeLimit);
// Register multipart with limits
fastify.register(multipart, {
  limits: {
    fileSize: fileSizeLimit,
  },
});

fastify.register(cors, {
  origin: "http://localhost:8080", // Your Vue.js frontend URL
  methods: ["GET", "POST"],
});

// Load the ONNX model
let session;

(async () => {
  try {
    session = await ort.InferenceSession.create(modelPath);
    fastify.log.info("ONNX Runtime session created");
  } catch (err) {
    fastify.log.error("Failed to create ONNX Runtime session:", err);
    process.exit(1);
  }
})();

// Define the prediction route
fastify.post("/predict", async (request, reply) => {
  try {
    // Handle the uploaded file
    const data = await request.file();

    if (!data) {
      throw new Error("No file uploaded");
    }

    const imageBuffer = await data.toBuffer();

    // Preprocess the image
    const inputTensor = await preprocessImage(imageBuffer);

    const inputName = session.inputNames[0];
    // Prepare model input
    const feeds = { [inputName]: inputTensor }; // 'input' should match your model's input name

    // Run inference
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];

    // Retrieve and send the prediction
    const outputTensor = results[outputName]; // 'output' should match your model's output name
    reply.send({ prediction: outputTensor.data });
  } catch (error) {
    fastify.log.error(error);
    reply.status(500).send({ error: "Prediction failed", details: error.message });
  }
});

// Start the Fastify server
const start = async () => {
  try {
    await fastify.listen({ port: serverPort, host: serverHost });
    fastify.log.info(`Fastify server running at http://${serverHost}:${serverPort}/`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
