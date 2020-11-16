import * as tf from "@tensorflow/tfjs";

async function setupModel() {
  // TODO:
    // local: "http://localhost:8000/src/model/model.json"
    // live host:

  const model_url = "../model/model.json";
  return tf.loadGraphModel(model_url);
}

export default setupModel;
