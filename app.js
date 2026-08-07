let session = null;
let coordMap = null;
let roleToColor = null;
let nameToGrade = null;
let boardImage = null;   // holds the loaded <img> once it's ready

function loadBoardImage() {
  boardImage = new Image();
  boardImage.src = "kilterboard.jpeg";
  boardImage.onload = function() {
    drawClimb([]);   // draw the board immediately, with zero holds on top yet
  };
}

async function init() {
  session = await ort.InferenceSession.create("KilterGen.onnx");

  const response = await fetch("KilterMap.json");
  const data = await response.json();
  coordMap = data.coord_map;
  roleToColor = data.role_to_color;
  nameToGrade = data.name_to_grade;

  loadBoardImage();

  document.getElementById("generate").addEventListener("click", onGenerate);
}

function buildCausalMask(len) { // define what we can talk to/ what we can see when we generate
  let arr = new Float32Array(len * len);
  for (let i = 0; i < len; i++) {   // we need to make the casual mask. here it should be that bottom triangle because
    for (let j = 0; j < len; j++) { // we want everything to see itself and what comes before it.
      if (j <= i) {
        arr[i* len + j] = 0;
      } else {
        arr[i * len + j] = -Infinity;
      }
    }
  }
  return arr;
}

async function generateClimb(angle, grade, nomatch) {
  angle = angle - (angle % 5);
  if (grade > 13) {
    grade = 13;
  }
  grade = nameToGrade["V" + grade];
  let token = [(angle / 5) + 1511, grade + 70 / 5 + 1511, nomatch + 70 / 5 + 1511 + 39];

  while (true) {
    const seqLen = token.length;
    const tokenTensor = new ort.Tensor("int64", BigInt64Array.from(token.map(BigInt)), [1, seqLen]);
    const attentionTensor = new ort.Tensor("float32", new Float32Array(seqLen).fill(1), [1, seqLen]);
    const casualTensor = new ort.Tensor("float32", buildCausalMask(seqLen), [seqLen, seqLen]);  // create input tensors
    const final = await session.run({ tokens: tokenTensor, attention: attentionTensor, casual: casualTensor })
    const logitsData = final.logits.data;
    const dims = final.logits.dims;
    const size = dims[2];
    const lastPosition = (seqLen - 1) * size;
    const lastLogits = logitsData.slice(lastPosition, lastPosition + size);

    // softmax like in python
    const maxVal = Math.max(...lastLogits);
    const exps = lastLogits.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((total, x) => total + x, 0);
    const probs = exps.map(x => x / sumExps);


    //Choose a random logit
    const r = Math.random();
    let cumulative = 0;
    let nextToken = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) {
        nextToken = i;
        break;
      }
    }

    token.push(nextToken);
    if (nextToken === 0 || token.length >= 150) {
      break;
    }
  }

  return token;
}

async function generateOneAttempt(angle, grade, nomatch) {
  let holds = await generateClimb(angle, grade, nomatch);
  holds.splice(0, 3);
  if (holds[holds.length - 1] === 0) {
    holds.pop();
  }
  let holdarr = [];
  for (let i = 0; i < holds.length - 1; i += 2) {
    const [x, y] = coordMap[holds[i]];
    const color = roleToColor[holds[i + 1]];
    const hold = [x, y, color];
    holdarr.push(hold)
  }
  return holdarr;
}

async function onGenerate() {
  const angle = parseInt(document.getElementById("angle").value);
  const grade = parseInt(document.getElementById("grade").value);
  const nomatch = parseInt(document.getElementById("Match").checked ? 1 : 0);

  let holdarr = [];
  for (let attempt = 0; attempt < 1000; attempt++) {
    const candidate = await generateOneAttempt(angle, grade, nomatch);

    let green = 0;
    let purple = 0;
    let foot = 0;
    for (const hold of candidate) {
      if (hold[2] === "start") {
        green++;
      }
      if (hold[2] === "finish") {
        purple++;
      }
      if (hold[2] === "foot") {
        foot++;
      }
    }

    if (green > 0 && green <= 2 && purple > 0 && purple <= 2 && foot > 0) {
      holdarr = candidate;
      break;
    }
  }

  drawClimb(holdarr);
}

function drawClimb(holds) {
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (boardImage) {
    ctx.drawImage(boardImage, 0, 0, canvas.width, canvas.height);
  }

  for (let i = 0; i < holds.length; i++) {
    const centerX = holds[i][0] * (canvas.width / 144);
    const centerY = canvas.height - holds[i][1] * (canvas.height / 156);
    let x = "";
    if (holds[i][2] == "foot") {
      x = "yellow";
    }
    if (holds[i][2] == "start") {
      x = "lime";
    }
    if (holds[i][2] == "finish") {
      x = "magenta";
    }
    if (holds[i][2] == "middle") {
      x = "cyan";
    }
    ctx.beginPath();
    ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
    ctx.strokeStyle = x;
    ctx.lineWidth = 3;
    ctx.stroke();
  }

}

init();
