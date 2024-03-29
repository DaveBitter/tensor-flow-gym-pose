import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

(async () => {
  const video = document.querySelector("[data-video]");
  const videoPlaceholder = document.querySelector("[data-video-placeholder]");
  const squadCounter = document.querySelector("[data-squad-counter]");
  const webcamSelect = document.querySelector("[data-webcam-select]");

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

  const connectors = [
    "ears",
    "shoulders",
    "left-shoulder-to-ear",
    "right-shoulder-to-ear",
    "left-shoulder-to-hip",
    "right-shoulder-to-hip",
    "left-upperarm",
    "right-upperarm",
    "left-lowerarm",
    "right-lowerarm",
    "hips",
    "right-upperbody",
    "left-upperbody",
    "left-upperleg",
    "right-upperleg",
    "left-lowerleg",
    "right-lowerleg",
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

  const lines = connectors.reduce(
    (acc, cur) => ({
      ...acc,
      [cur]: document.querySelector(`[data-line-${cur}]`),
    }),
    {}
  );

  await tf.ready();

  const detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet
  );

  const state = {
    prevHipY: null,
    curHipY: null,
    baseHipY: null,
    squatPosition: null,
    eligbleForPoint: false,
    squadCount: 0,
    hasFilledWebcamOptions: false,
  };

  const loadVideoStream = (stream) =>
    new Promise((resolve) => {
      video.srcObject = stream;
      videoPlaceholder.srcObject = stream;
      videoPlaceholder.addEventListener("loadedmetadata", resolve);
    });

  const onUserMedia = (stream) => {
    return new Promise(async (resolve) => {
      await loadVideoStream(stream);

      video.play();
      videoPlaceholder.play();

      resolve();
    });
  };

  const initWebcamStreams = async (deviceIdToUse) => {
    const mediaDevices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = mediaDevices.filter(
      (mediaDevice) => mediaDevice.kind === "videoinput"
    );

    !state.hasFilledWebcamOptions &&
      videoDevices.forEach((videoDevice, index) => {
        const option = document.createElement("option");
        const label = videoDevice.label.split(" (")[0] || `Camera ${index++}`;
        const textNode = document.createTextNode(label);

        option.value = videoDevice.deviceId;
        option.appendChild(textNode);
        webcamSelect.appendChild(option);

        state.hasFilledWebcamOptions = true;
      });

    const cachedVideoDeviceId = localStorage.getItem("selectedVideoDeviceId");

    const selectedDeviceId =
      deviceIdToUse || cachedVideoDeviceId || videoDevices[0].deviceId;
    webcamSelect.value = selectedDeviceId;

    localStorage.setItem("selectedVideoDeviceId", selectedDeviceId);

    await navigator.mediaDevices
      .getUserMedia({
        video: {
          deviceId: selectedDeviceId,
        },
        audio: false,
      })
      .then(onUserMedia)
      .catch(alert);
  };

  initWebcamStreams();

  webcamSelect.addEventListener("change", ({ target }) => {
    return initWebcamStreams(target.value);
  });

  // const mediaDevices = navigator.mediaDevices;
  // mediaDevices
  //   .getUserMedia({
  //     video: true,
  //     audio: false,
  //   })
  //   .then((stream) => {
  //     // Changing the source of video to current stream.
  //     video.srcObject = stream;
  //     videoPlaceholder.srcObject = stream;
  //     video.addEventListener("loadedmetadata", () => {
  //       video.play();
  //     });
  //   });

  const drawLine = (left, right, line) => {
    const box = {
      x: parseInt(left.x < right.x ? left.x : right.x),
      y: parseInt(left.y < right.y ? left.y : right.y),
      width: parseInt(left.x < right.x ? right.x - left.x : left.x - right.x),
      height: parseInt(left.y < right.y ? right.y - left.y : left.y - right.y),
    };

    let lineOrientation;
    switch (true) {
      case left.x < right.x && left.y > right.y:
        lineOrientation = "top-left-to-bottom-right";
        break;
      case left.x < right.x && left.y < right.y:
        lineOrientation = "bottom-left-to-top-right";
        break;
      case left.x > right.x && left.y > right.y:
        lineOrientation = "bottom-left-to-top-right";
        break;
      case left.x > right.x && left.y < right.y:
        lineOrientation = "top-left-to-bottom-right";
        break;

      case right.x < left.x && right.y > left.y:
        lineOrientation = "top-left-to-bottom-right";
        break;
      case right.x < left.x && right.y < left.y:
        lineOrientation = "bottom-left-to-top-right";
        break;
      case right.x > left.x && right.y > left.y:
        lineOrientation = "bottom-left-to-top-right";
        break;
      case right.x > left.x && right.y < left.y:
        lineOrientation = "top-left-to-bottom-right";
        break;
      default:
        break;
    }

    line.dataset.lineOrientation = lineOrientation;
    line.dataset.active = `${left.score > 0.5 && right.score > 0.5}`;
    line.style.transform = `translate(${box.x}px, ${box.y}px)`;
    line.style.maxWidth = `${box.width}px`;
    line.style.maxHeight = `${box.height}px`;
  };

  setInterval(async () => {
    const [pose] = await detector.estimatePoses(video);
    const { score, keypoints } = pose;
    outputs.overall.innerHTML = `<span data-score=${
      score > 0.5 ? "good" : "bad"
    }>${parseInt(score * 100)}% </span>`;

    const values = {};
    keypoints.forEach(({ name, x, y, score }) => {
      const videoWidthByAspectRatio =
        window.innerHeight *
        (videoPlaceholder.scrollWidth / videoPlaceholder.scrollHeight);
      const videoHeightByAspectRatio =
        window.innerWidth *
        (videoPlaceholder.scrollHeight / videoPlaceholder.scrollWidth);

      let factorX;
      let factorY;

      if (window.innerHeight > window.innerWidth) {
        factorY = window.innerHeight / videoPlaceholder.scrollHeight;
        factorX = videoWidthByAspectRatio / videoPlaceholder.scrollWidth;
      } else {
        factorY = videoHeightByAspectRatio / videoPlaceholder.scrollHeight;
        factorX = window.innerWidth / videoPlaceholder.scrollWidth;
      }

      const translateX = factorX * x;
      const translateY = factorY * y;
      values[name] = { x: translateX, y: translateY, score };

      outputs[name.replace("_", "-")].innerHTML = `${parseInt(
        translateX
      )}, ${parseInt(translateY)}, <span data-score=${
        score > 0.5 ? "good" : "bad"
      }>${parseInt(score * 100)}% </span>`;

      markers[name.replace("_", "-")].dataset.active = `${score > 0.5}`;
      markers[name.replace("_", "-")].style.transform = `translate(${
        translateX - 15
      }px, ${translateY - 15}px)`;

      if (name === "left_hip") {
        if (!state.baseHipY) {
          state.baseHipY = y;
          state.squatPosition = "up";
        }

        if (y > state.baseHipY * 1.2) {
          state.squatPosition = "down";
          state.eligbleForPoint = true;
        } else if (y > state.baseHipY) {
          state.squatPosition = "between";
        } else {
          state.squatPosition = "up";

          if (state.eligbleForPoint) {
            state.squadCount += 1;
            state.eligbleForPoint = false;
          }
        }

        state.curHipY = y;

        squadCounter.innerHTML = state.squadCount;
      }
    });

    const {
      left_ear,
      right_ear,
      left_shoulder,
      right_shoulder,
      left_elbow,
      left_wrist,
      right_elbow,
      right_wrist,
      left_hip,
      left_knee,
      right_hip,
      right_knee,
      left_ankle,
      right_ankle,
    } = values;

    drawLine(left_ear, right_ear, lines.ears);
    drawLine(left_shoulder, right_shoulder, lines.shoulders);
    drawLine(left_shoulder, left_ear, lines["left-shoulder-to-ear"]);
    drawLine(right_shoulder, right_ear, lines["right-shoulder-to-ear"]);
    drawLine(left_shoulder, left_hip, lines["left-shoulder-to-hip"]);
    drawLine(right_shoulder, right_hip, lines["right-shoulder-to-hip"]);
    drawLine(left_shoulder, left_elbow, lines["left-upperarm"]);
    drawLine(right_shoulder, right_elbow, lines["right-upperarm"]);
    drawLine(left_elbow, left_wrist, lines["left-lowerarm"]);
    drawLine(right_elbow, right_wrist, lines["right-lowerarm"]);
    drawLine(left_hip, left_shoulder, lines["left-upperbody"]);
    drawLine(left_hip, right_hip, lines.hips);
    drawLine(left_hip, left_knee, lines["left-upperleg"]);
    drawLine(right_hip, right_shoulder, lines["right-upperbody"]);
    drawLine(right_hip, right_knee, lines["right-upperleg"]);
    drawLine(left_knee, left_ankle, lines["left-lowerleg"]);
    drawLine(right_knee, right_ankle, lines["right-lowerleg"]);
  }, 1000 / 60);
})();
