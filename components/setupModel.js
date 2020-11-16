import * as tf from "@tensorflow/tfjs";

const base_url = process.env.BASE_URL;
async function setupModel() {
  let model_url = `${base_url}/model/model.json`;
  if (process.env.NODE_ENV == "development") {
    model_url = "http://localhost:8000/public/model/model.json";
  }
  return tf.loadGraphModel(model_url);
}

export default setupModel;
