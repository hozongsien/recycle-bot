import * as tf from "@tensorflow/tfjs";
import Head from "next/head";
import React, { useRef, useState, useEffect } from "react";
import {
  setupCamera,
  startVideo,
  VIDEO_WIDTH_PIXELS,
  VIDEO_HEIGHT_PIXELS,
} from "../components/camera";
import {
  setupModel,
  warmUpModel,
  getPrediction,
  getTopKClasses,
} from "../components/setupModel";

export default function Home() {
  const base_url = process.env.BASE_URL;
  const videoRef = useRef();
  const [loadedModel, setModel] = useState();
  const [prediction, setPrediction] = useState();
  let requestAnimationFrameId = 0;

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

      return getPrediction(model, pixelsCropped);
    });

    const topK = getTopKClasses(result, 1);
    setPrediction(topK[0].label);
    requestAnimationFrameId = requestAnimationFrame(() =>
      predict(model, videoRef)
    );
  };

  const startPrediction = async () => {
    await warmUpModel(loadedModel);
    await predict(loadedModel, videoRef);
  };

  const loadAssets = async () => {
    await tf.ready();
    await setupCamera(videoRef);
    const model = await setupModel();
    setModel(model);
  };

  useEffect(() => {
    loadAssets();
  }, []);

  useEffect(() => {
    if (!loadedModel) {
      return;
    }

    startPrediction();
  }, [loadedModel]);

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
