import fs from "fs";

export function getTopNPredictions(predictionObj, n) {
  const predictionsArray = Object.entries(predictionObj).map(([key, value]) => ({
    index: Number(key),
    score: value,
  }));

  predictionsArray.sort((a, b) => b.score - a.score);
  const topPredictionsArray = predictionsArray.slice(0, n);

  // Create a Map to store predictions in order
  const topPredictionsMap = new Map();
  topPredictionsArray.forEach(({ index, score }) => {
    topPredictionsMap.set(index, score);
  });

  return topPredictionsMap;
}

export function loadMapping(jsonFilePath) {
  try {
    const data = fs.readFileSync(jsonFilePath, "utf8");
    return JSON.parse(data);
  } catch (err) {
    console.error("Error loading JSON file:", err);
    return null;
  }
}

export function getNamesFromTopPredictions(topPredictions, mapping) {
  const names = [];
  for (const [index] of topPredictions) {
    names.push(mapping[index] || `Unknown(${index})`);
  }
  return names;
}
