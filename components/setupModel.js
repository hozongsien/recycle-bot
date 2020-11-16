import * as tf from "@tensorflow/tfjs";

async function setupModel() {
  // TODO:
    // local: "http://localhost:8000/src/public/model/model.json"
    // live: "model/model.json"

  const model_url = "/model/model.json";
  return tf.loadGraphModel(model_url);
}

export default setupModel;
