const VIDEO_WIDTH_PIXELS = 384;
const VIDEO_HEIGHT_PIXELS = 512;

export { VIDEO_WIDTH_PIXELS, VIDEO_HEIGHT_PIXELS };

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

const setupVideoDimensions = (videoRef) => {
  const height = videoRef.current.height;
  const width = videoRef.current.width;
  const aspectRatio = width / height;

  videoRef.current.width = VIDEO_WIDTH_PIXELS;
  videoRef.current.height = VIDEO_HEIGHT_PIXELS;
};

const startVideo = (videoRef) => {
  videoRef.current.play();
};

export { setupCamera, setupVideoDimensions, startVideo };
