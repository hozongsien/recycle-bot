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
      height: 512,
      width: 384,
    },
  });
  videoRef.current.srcObject = stream;

  return new Promise((resolve) => {
    videoRef.current.onloadedmetadata = () => {
      resolve(videoRef.current);
    };
  });
};

export default setupCamera;
