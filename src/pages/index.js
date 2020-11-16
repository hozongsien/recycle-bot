import * as tf from "@tensorflow/tfjs";
import Head from "next/head";
import React, { useRef, useState, useEffect, useContext } from "react";
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
  let requestAnimationFrameId = 0;

  const predict = async (model, video) => {
    const result = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(video);
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

    const probs = result.dataSync();
    const label = tf.argMax(probs).dataSync();
    setPrediction(CLASSES[label])
    result.dispose()
    requestAnimationFrameId = requestAnimationFrame(() => predict(model, video));
  };
  const warmUpModel = async (model) => {
    model.predict(tf.zeros([1, VIDEO_HEIGHT_PIXELS, VIDEO_WIDTH_PIXELS, 3]))
  }

  const enablePrediction = async () => {
    await tf.ready();
    // const model = await setupModel()
    // await warmUpModel(model)
    const video = await setupCamera(videoRef)
    // await predict(model, video)
  };

  useEffect(() => {
    console.log('mount')
    enablePrediction();

    return () => {
      console.log("unmount");
    };
  }, []);


  return (
    <div className="container">
      <Head>
        <title>Recycle Bot</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1 className="title">Recycle Bot</h1>
        <p className="description">Classify waste.</p>
        <p className="description">{prediction}</p>
        <div className="frame">
          <video className="video" autoPlay playsInline muted ref={videoRef} />
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

        a {
          color: inherit;
          text-decoration: none;
        }

        .title a {
          color: #0070f3;
          text-decoration: none;
        }

        .title a:hover,
        .title a:focus,
        .title a:active {
          text-decoration: underline;
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
