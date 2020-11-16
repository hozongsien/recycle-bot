import * as tf from "@tensorflow/tfjs";
import Head from "next/head";
import React, { useRef, useState, useEffect } from "react";
import setupCamera from "../components/setupCamera";
import setupModel from "../components/setupModel";

export default function Home() {
  const VIDEO_HEIGHT_PIXELS = 512;
  const VIDEO_WIDTH_PIXELS = 384;
  const videoRef = useRef();
  const [mod, setMod] = useState();
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

    const topK = getTopKClasses(result, 1);
    setPrediction(topK[0].label);
    requestAnimationFrameId = requestAnimationFrame(() =>
      predict(model, videoRef)
    );
  };

  const setupVideoDimensions = (videoRef) => {
    const height = videoRef.current.height;
    const width = videoRef.current.width;
    const aspectRatio = width / height;

    videoRef.current.width = VIDEO_WIDTH_PIXELS;
    videoRef.current.height = VIDEO_HEIGHT_PIXELS;
  };

  const startVideo = (video) => {
    video.current.play();
  };
  const warmUpModel = async (model) => {
    model.predict(tf.zeros([1, VIDEO_HEIGHT_PIXELS, VIDEO_WIDTH_PIXELS, 3]));
  };
  const enablePrediction = async () => {
    await tf.ready();

    await setupCamera(videoRef);
    // setupVideoDimensions(videoRef);
    startVideo(videoRef);
    const model = await setupModel();
    await warmUpModel(model);
    setMod(model);

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
        <meta name="viewport" content="initial-scale=1.0, width=device-width" />
      </Head>

      <main>
        <div className="title">Recycle Bot</div>
        <div className="description">Classifies waste.</div>
        <div className="description">
          {prediction ? prediction : "making prediction"}
        </div>
        <video className="video" autoPlay playsInline muted ref={videoRef} />
      </main>

      <footer>
        <p>Ho Zong Sien</p>
      </footer>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        .video {
          width: 100%;
        }

        main {
          padding: 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          width: 100%;
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

        .logo {
          height: 1em;
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
