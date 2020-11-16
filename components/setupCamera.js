const setupCamera = async (videoRef) => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  videoRef.current.srcObject = stream;

  return new Promise((resolve) => {
    videoRef.current.onloadedmetadata = () => {
      resolve();
    };
  });
};

export default setupCamera;
