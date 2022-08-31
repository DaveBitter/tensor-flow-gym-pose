import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

(async () => {
  const video = document.querySelector("[data-video]");

  const detections = [
    "overall",
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
  ];

  const outputs = detections.reduce(
    (acc, cur) => ({
      ...acc,
      [cur]: document.querySelector(`[data-detection-output-${cur}]`),
    }),
    {}
  );

  const markers = detections.reduce(
    (acc, cur) => ({
      ...acc,
      [cur]: document.querySelector(`[data-marker-${cur}]`),
    }),
    {}
  );

  await tf.ready();

  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet
  );

  const mediaDevices = navigator.mediaDevices;
  mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      // Changing the source of video to current stream.
      video.srcObject = stream;
      video.addEventListener("loadedmetadata", () => {
        video.play();
      });
    });

  setInterval(async () => {
    const [pose] = await detector.estimatePoses(video);
    const { score, keypoints } = pose;
    outputs.overall.innerHTML = `<span data-score=${
      score > 0.5 ? "good" : "bad"
    }>${parseInt(score * 100)}% </span>`;

    keypoints.forEach(({ name, x, y, score }) => {
      const translateX = (x / 640) * window.innerWidth;
      const translateY = (y / 480) * window.innerHeight;
      outputs[name.replace("_", "-")].innerHTML = `${parseInt(x)}, ${parseInt(
        y
      )}, <span data-score=${score > 0.5 ? "good" : "bad"}>${parseInt(
        score * 100
      )}% </span>`;

      markers[name.replace("_", "-")].dataset.active = `${score > 0.5}`;
      markers[name.replace("_", "-")].style.transform = `translate(${
        translateX - 15
      }px, ${translateY - 15}px)`;
    });
  }, 1000 / 60);
})();
