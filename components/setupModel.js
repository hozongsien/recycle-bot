import * as tf from "@tensorflow/tfjs";
import { VIDEO_WIDTH_PIXELS, VIDEO_HEIGHT_PIXELS } from "../components/camera";

const base_url = process.env.BASE_URL;
const isProd = process.env.NODE_ENV == "production";
const CLASSES = {
  0: "cardboard",
  1: "glass",
  2: "metal",
  3: "paper",
  4: "plastic",
  5: "trash",
};

const setupModel = async () => {
  let model_url = `${base_url}/model/model.json`;
  if (!isProd) {
    model_url = "http://localhost:8000/public/model/model.json";
  }
  return tf.loadGraphModel(model_url);
};

const warmUpModel = async (model) => {
  model.predict(tf.zeros([1, VIDEO_HEIGHT_PIXELS, VIDEO_WIDTH_PIXELS, 3]));
};

const getTopKClasses = (predictions, topK) => {
  const values = predictions.dataSync();
  predictions.dispose();

  let predictionList = [];
  for (let i = 0; i < values.length; i++) {
    predictionList.push({ value: values[i], index: i });
  }
  predictionList = predictionList
    .sort((a, b) => {
      return b.value - a.value;
    })
    .slice(0, topK);

  return predictionList.map((x) => {
    return { label: CLASSES[x.index], value: x.value };
  });
};

const getPrediction = (model, input) => {
  // TODO: preprocess
  const inputExp = input.expandDims(0);
  const inputExpCas = tf.cast(inputExp, "float32");
  return model.execute(inputExpCas);
}

export { setupModel, warmUpModel, getTopKClasses, getPrediction };
