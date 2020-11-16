import * as tf from "@tensorflow/tfjs";
import Head from "next/head";
import React, { useRef, useState, useEffect } from "react";
import setupCamera from "../components/setupCamera";
import setupModel from "../components/setupModel";

export default function Home() {
  const VIDEO_HEIGHT_PIXELS = 512;
  const VIDEO_WIDTH_PIXELS = 384;
  const videoRef = useRef();
  const [prediction, setPrediction] = useState();
  const CLASSES = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
  };
  const base_url = process.env.BASE_URL;
  let requestAnimationFrameId = 0;
  
  const getTopKClasses = (predictions, topK) => {
    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList.sort((a, b) => {
      return b.value - a.value;
    }).slice(0, topK);

    return predictionList.map(x => {
      return {label: CLASSES[x.index], value: x.value};
    });
  }

  const predict = async (model, videoRef) => {
    const result = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(videoRef.current);
      const centerHeight = pixels.shape[0] / 2;
      const beginHeight = centerHeight - VIDEO_HEIGHT_PIXELS / 2;
      const centerWidth = pixels.shape[1] / 2;
      const beginWidth = centerWidth - VIDEO_WIDTH_PIXELS / 2;
      const pixelsCropped = pixels.slice(
        [beginHeight, beginWidth, 0],
        [VIDEO_HEIGHT_PIXELS, VIDEO_WIDTH_PIXELS, 3]
      );

      const pixelsCrpExp = pixelsCropped.expandDims(0);
      const PixelsCrpExpSc = tf.cast(pixelsCrpExp, "float32");
      return model.execute(PixelsCrpExpSc);
    });

    const topK = getTopKClasses(result, 1)
    setPrediction(topK[0].label);
    requestAnimationFrameId = requestAnimationFrame(() =>
      predict(model, videoRef)
    );
  };

  const startVdieo = async (video) => {
    video.current.play();
  };
  const warmUpModel = async (model) => {
    model.predict(tf.zeros([1, VIDEO_HEIGHT_PIXELS, VIDEO_WIDTH_PIXELS, 3]));
  };
  const enablePrediction = async () => {
    await tf.ready();

    const video = await setupCamera(videoRef);
    await startVdieo(videoRef);
    const model = await setupModel();
    await warmUpModel(model);

    await predict(model, videoRef);
  };

  useEffect(() => {
    console.log("mount");
    enablePrediction();

    return () => {
      console.log("unmount");
    };
  }, []);

  return (
    <div className="container">
      <Head>
        <title>Recycle Bot</title>
        <link rel="icon" href={`${base_url}/favicon.ico`} />
      </Head>

      <main>
        <h1 className="title">Recycle Bot</h1>
        <p className="description">Classify waste.</p>
        <img src={`${base_url}/images/tfjs.png`} alt="tfjs" width={200} />
        <p className="description">
          {prediction ? prediction : "loading model"}
        </p>
        <div className="frame">
          <video className="video" playsInline muted ref={videoRef} />
        </div>
      </main>

      <footer>
        <p>Ho Zong Sien</p>
      </footer>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0 0.5rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        .frame {
          border: solid;
        }

        main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        footer {
          width: 100%;
          height: 100px;
          border-top: 1px solid #eaeaea;
          display: flex;
          justify-content: center;
          align-items: center;
        }

        footer a {
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 4rem;
        }

        .title,
        .description {
          text-align: center;
        }

        .description {
          line-height: 1.5;
          font-size: 1.5rem;
        }

        .grid {
          display: flex;
          align-items: center;
          justify-content: center;
          flex-wrap: wrap;

          max-width: 800px;
          margin-top: 3rem;
        }

        .logo {
          height: 1em;
        }

        @media (max-width: 600px) {
          .grid {
            width: 100%;
            flex-direction: column;
          }
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }

        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  );
}
