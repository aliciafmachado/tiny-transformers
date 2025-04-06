/* Copyright 2023 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*

Tiny Worlds, run with (gtensor-based) transformers.

TODO: add yargs so this is a real command line tool example.

Run:
  npx ts-node src/lib/seqtasks/experiment_with_dynamic_residuals.ts
*/

import * as tf from '@tensorflow/tfjs-node';
import * as Plot from '@observablehq/plot';
import sharp from "sharp";
import fs from "fs";
import { JSDOM } from "jsdom";
import { GTensor } from "../gtensor/gtensor";


import {
  TransformerComputation,
  lastTokenLogits,
  allPastTokensCrossEntropyLossWithIntegerLabels,
  allPastTokensLogits,
} from '../transformer/common_transformer';
import {
  TransformerParamLayerSpec,
  TransformerParamSpec,
  TransformerParams,
  TransformerModel,
  Config,
  initDecoderParams,
  computeTransformer,
} from '../transformer/gpt2';
import { TinyWorldTask, TinyWorldTaskConfig, defaultTinyWorldTaskConfig } from './tiny_worlds';
import {
  strSeqPrepFn,
  singleNextTokenIdxOutputPrepFn,
  expectedOutputSeqPrepFn,
  prepareBasicTaskTokenRep,
  BasicTaskTokenRep,
} from '../tokens/token_gemb';
import { varifyParams, listifyVarParams } from '../gtensor/params';
import { RandomStream, makeRandomStream } from '../random/random';
import { isNumber } from 'underscore';

const tfjsBackendName = tf.getBackend();
console.log('tfjs backend:', tfjsBackendName);

const printEveryNBatches = 50;

interface Metric {
  loss: number;
  step: number;
  accuracy?: number;
  klDivergence?: number;
  alphaFirstParams?: number[];
  alphaSecondParams?: number[];
}

interface EvalMetric {
  step: number;
  accuracy: number;
}

const availableMetrics: string[] = ["loss", "accuracy", "klDivergence", "alphaFirstParams", "alphaSecondParams"];
const availableEvalMetrics: string[] = ["accuracy"];

// TODO: this should be a class with default values. 
interface ExperimentConfig {
  name: string,
  useResiduals: boolean,
  useAlphaParams: boolean,
  learningRate: number,
  nIterations: number,
  nBatchSize: number,
  unfreezeEveryNSteps: number,
  nHeads: number;
  startFreezingAtIndex: number,
  seed: number,
}

const MAXNUMBER = 1000;

function defaultConfigs(config: Partial<ExperimentConfig> = {}): ExperimentConfig {
  return {
    name: config.name ?? "defaultExperiment",
    useResiduals: config.useResiduals ?? true,
    useAlphaParams: config.useAlphaParams ?? false,
    learningRate: config.learningRate ?? 0.001,
    nIterations: config.nIterations ?? 100,
    nBatchSize: config.nBatchSize ?? 64,
    unfreezeEveryNSteps: config.unfreezeEveryNSteps ?? MAXNUMBER,
    nHeads: config.nHeads ?? 3,
    startFreezingAtIndex: config.startFreezingAtIndex ?? MAXNUMBER,
    seed: config.seed ?? 0,
  };
}

function getTaskConfig(): TinyWorldTaskConfig {
  const taskConfig: TinyWorldTaskConfig = {
    ...defaultTinyWorldTaskConfig,
    maxInputLen: 10,
    maxOutputLen: 1,
  };
  return taskConfig;
}

function initTransformerConfig(baseVocab: string[], nHeads: number = 6, alphaParams: boolean = false,
  residuals: boolean = true, seed: number,
): Config {
  // Set dummy transformer for testing.
  const embeddingSize = 16; // 64 * 12 originally
  const posEmbeddings = 16;
  const layerConfig: TransformerParamLayerSpec = {
    nHeads: nHeads,
    layerNormPreAttention: false,
    layerNormHeadsProjection: false,
    addLayerNormBias: false,
    computeSpec: { residuals: residuals, dropoutRate: 0, layerNormEpsilon: 1e-5 },
  };
  const spec: TransformerParamSpec = {
    inputRep: embeddingSize * nHeads,
    kqvRep: embeddingSize,
    layers: Array(nHeads).fill(layerConfig),
    computeSpec: {
      dropoutRate: 0.0,
      layerNormEpsilon: 1e-5
    },
    posEncodingSeqLength: posEmbeddings,
    layerNorm: false,
    addLayerNormBias: false,
    addPosEmbeddings: true,
    addAlphaParameter: alphaParams,
  };
  const config: Config = {
    id: 'MiniGPT2',
    kind: 'Transformer',
    spec: spec,
    tokenRep: prepareBasicTaskTokenRep(baseVocab),
    init: {
      stddev: 0.05, // default
      mean: 0,
      seed: seed,
    },
  };
  return config;
}

type Batch = {
  batchId: number;
  inputs: string[][];
  outputs: string[][];
};

function* batchGenerator(
  task: TinyWorldTask,
  batchNum: number,
  batchSize: number
): Iterable<Batch> {
  for (let batchId = 1; batchId <= batchNum; batchId += 1) {
    let batchOriginal = task.exampleIter.takeOutN(batchSize);
    let inputs = batchOriginal.map((example) => example.input);
    let outputs = batchOriginal.map((example) => example.output);
    yield { batchId, inputs, outputs };
  }
}

function computeModelAndPrepareOutput(
  model: {
    config: {
      spec: TransformerParamSpec;
      tokenRep: BasicTaskTokenRep;
    }
    params: TransformerParams;
  },
  randomStream: RandomStream,
  batchInput: string[][],
  batchOutput: string[][],
): { "target": GTensor<"batch" | "pos">, "computation": TransformerComputation } {
  const maxInputLength = batchInput.reduce(
    (max, curInput) => (max >= curInput.length ? max : curInput.length),
    0,
  );
  const gtensorInputs = strSeqPrepFn(model, batchInput, { maxInputLength });
  const computation: TransformerComputation = computeTransformer(
    model,
    gtensorInputs,
    randomStream
  );
  const targetTokens = expectedOutputSeqPrepFn(model, batchInput, batchOutput);
  return { "target": targetTokens, "computation": computation };
}


function computeAccuracy(
  model: {
    config: {
      spec: TransformerParamSpec;
      tokenRep: BasicTaskTokenRep;
    }
    params: TransformerParams;
  },
  randomStream: RandomStream,
  // batchId: number,
  batchInput: string[][],
  batchOutput: string[][],
  // thisExperimentMetrics: Metric[],
) {
  // Disable dropout.
  const dropout = model.config.spec.computeSpec.dropoutRate;
  model.config.spec.computeSpec.dropoutRate = 0;
  for (const layer of model.config.spec.layers) {
    layer.computeSpec.dropoutRate = 0;
  }
  // Regular computation.
  const targetTokensAndComputation = computeModelAndPrepareOutput(model, randomStream, batchInput, batchOutput);
  const expectedOutputSeq = new GTensor(
    tf.tensor(batchOutput.map((outputToken) => model.config.tokenRep.tokenToIdx[outputToken[0]]), undefined, 'int32'), ["batch"])
  const classifiedToken = lastTokenLogits(model, targetTokensAndComputation.computation).argMax("tokenId");
  const accuracy = classifiedToken.pointwiseEqual(
    expectedOutputSeq).sumOverDims(["batch"]).tensor.div(tf.scalar(expectedOutputSeq.dim.batch.size));
  // thisExperimentMetrics[-1].accuracy =  accuracy.asScalar().arraySync();

  // Add dropout back.
  // Note: we are supposing that the dropout is the same everywhere.
  model.config.spec.computeSpec.dropoutRate = dropout;
  for (const layer of model.config.spec.layers) {
    layer.computeSpec.dropoutRate = dropout;
  }

  return accuracy.asScalar().arraySync();
}

function computeLoss(
  model: {
    config: {
      spec: TransformerParamSpec;
      tokenRep: BasicTaskTokenRep;
    }
    params: TransformerParams;
  },
  randomStream: RandomStream,
  batchId: number,
  batchInput: string[][],
  batchOutput: string[][],
  thisExperimentMetrics: Metric[],
): tf.Scalar {
  const targetTokensAndComputation = computeModelAndPrepareOutput(model, randomStream, batchInput, batchOutput);
  const entropyLoss: tf.Scalar = allPastTokensCrossEntropyLossWithIntegerLabels(
    model, targetTokensAndComputation.computation, targetTokensAndComputation.target);
  let metric: Metric = { "loss": entropyLoss.arraySync(), "step": batchId };

  const alphaFirsts = model.params.layers.map((g) => g.alphaParams?.alphaFirst.tensor.asScalar().arraySync());
  const alphaSeconds = model.params.layers.map((g) => g.alphaParams?.alphaSecond.tensor.asScalar().arraySync());
  if (alphaFirsts[0] !== undefined) {
    metric.alphaFirstParams = alphaFirsts as number[];
  }
  if (alphaSeconds[0] !== undefined) {
    metric.alphaSecondParams = alphaSeconds as number[];
  }

  if ((batchId) % printEveryNBatches === 0) {
    console.log(
      `batch: ${batchId} `.padEnd(15) +
      ('entropyLoss: ' + entropyLoss.arraySync().toFixed(8)).padEnd(25)
    );
    if (alphaFirsts[0] !== undefined) {
      console.log(
        `alphaFirst: ${alphaFirsts.map((g) => g?.toFixed(8))}`
      )
    }
    if (alphaSeconds[0] !== undefined) {
      console.log(
        `alphaSecond: ${alphaSeconds.map((g) => g?.toFixed(8))}`
      )
    }
  }

  metric.accuracy = computeAccuracy(model, randomStream, batchInput, batchOutput);
  // Store loss for plotting.
  thisExperimentMetrics.push(metric);
  return entropyLoss;
}

function initParametersTrainableButAlphaFrom(transformerParams: TransformerParams, start_alpha: number) {
  // Interval of alphas that should not be trainable.
  for (let i = start_alpha; i < transformerParams.layers.length; i++) {
    const layer = transformerParams.layers[i];
    if (layer.alphaParams) {
      (layer.alphaParams.alphaFirst.tensor as tf.Variable).trainable = false;
      (layer.alphaParams.alphaSecond.tensor as tf.Variable).trainable = false;
    }
  };
}

function unfreezeAlphaParamsAt(transformerParams: TransformerParams, index: number) {
  const layer = transformerParams.layers[index];
  if (layer.alphaParams) {
    (layer.alphaParams.alphaFirst.tensor as tf.Variable).trainable = true;
    (layer.alphaParams.alphaSecond.tensor as tf.Variable).trainable = true;
  }
}

function run(experimentConfig: ExperimentConfig) {
  // initialize metrics
  let thisExperimentMetrics: Metric[] = [];
  // define task
  const trainTaskConfig = getTaskConfig();
  const trainTask = new TinyWorldTask(trainTaskConfig);

  // define vocab & decoder
  const Config = initTransformerConfig(trainTask.baseVocab, experimentConfig.nHeads, experimentConfig.useAlphaParams, experimentConfig.useResiduals,
    experimentConfig.seed
  );
  const decoderParams = varifyParams(initDecoderParams(Config));
  const model: TransformerModel = {
    config: Config,
    params: decoderParams as TransformerParams,
  };
  const randomStream = makeRandomStream(experimentConfig.seed);

  // By manipulating decoderParams, you can selectively limit what parameters
  // get tuned. By manipulating, we mean changing the trainable state to false.
  initParametersTrainableButAlphaFrom(decoderParams as TransformerParams, experimentConfig.startFreezingAtIndex);
  let paramsList = listifyVarParams(decoderParams).map((g) => g.variable);
  let unfreezeId = experimentConfig.startFreezingAtIndex;

  {
    // train with optimization
    const batchNum: number = experimentConfig.nIterations;
    const batchSize: number = experimentConfig.nBatchSize;

    let optimizer = tf.train.adam(experimentConfig.learningRate);
    for (let batch of batchGenerator(trainTask, batchNum, batchSize)) {
      let { batchId, inputs, outputs } = batch;
      optimizer.minimize(
        () => computeLoss(model, randomStream, batchId, inputs, outputs, thisExperimentMetrics),
        false,
        paramsList,
      );

      // if ((batchId) % printEveryNBatches === 0) {
      //   // Validation.
      //   let batchOriginal = trainTask.exampleIter.takeOutN(batchSize);
      //   let inputs = batchOriginal.map((example) => example.input);
      //   let outputs = batchOriginal.map((example) => example.output);

      // }

      batchId += 1;

      if ((batchId) % experimentConfig.unfreezeEveryNSteps == 0 && unfreezeId < decoderParams.layers.length) {
        unfreezeAlphaParamsAt(decoderParams, unfreezeId);
        paramsList = listifyVarParams(decoderParams).map((g) => g.variable);
        unfreezeId += 1;
      }
    }
    optimizer.dispose();
  }

  {
    // infer
    const inferSteps = 5;
    const inferTaskConfig = { ...getTaskConfig(), maxOutputLen: inferSteps };
    const inferTask = new TinyWorldTask(inferTaskConfig);

    const batchOriginal = inferTask.exampleIter.takeOutN(1);

    batchOriginal.forEach((e) =>
      console.log(`(${e.id}) ${e.input.join('')} ---> ${e.output.join('')}`)
    );

    const batchInputAll = batchOriginal.map((example) => example.input);
    const batchOutputAll = batchOriginal.map((example) => example.output);
    let batchInput = batchInputAll;
    // Make the batch output only have a single next token.
    let batchOutput = batchOutputAll.map((subarr) => subarr.slice(0, 1));

    // for (let inferStep = 0; inferStep < inferSteps; inferStep += 1) {
    const inferStep = 0;
    const spec = Config.spec;

    const maxInputLength = batchInput.reduce(
      (max, curInput) => (max >= curInput.length ? max : curInput.length),
      0,
    );
    const gtensorInputs = strSeqPrepFn(model, batchInput, { maxInputLength });
    const computation: TransformerComputation = computeTransformer(
      model,
      gtensorInputs,
      randomStream
    );
    //
    const singleNextTokenIdx = singleNextTokenIdxOutputPrepFn(model, batchOutput);
    // [0] to look at only the first example in batch.
    const singleNextTokenIdxArrayData = (singleNextTokenIdx.tensor.arraySync() as number[])[0];
    const logits = lastTokenLogits(model, computation);
    // TODO: tensor.arraySync() doesn't provide any guarentee for the ordering of outputs,
    // we need to use the right gtensor functions to get the output we want...
    // [0] to look at only the first example in batch.
    const logitsArr = (logits.tensor.arraySync() as number[][])[0];
    let probs = logits.softmax('tokenId');
    // [0] to look at only the first example in batch.
    let probsArrayData = (probs.tensor.arraySync() as number[][])[0];

    // Create a sorted table of information for each token.
    const possibleTokenTable = probsArrayData.map((prob, i) => {
      return { str: model.config.tokenRep.tokens[i], tokenId: i, prob: prob, logit: logitsArr[i] };
    });
    possibleTokenTable.sort((a, b) => b.prob - a.prob);

    console.log('Inference Step:', inferStep);
    console.log('Context:', batchInput[0].join(''));
    // console.log('Target Output:', batchOutput[0].join(''));
    console.log('Target next token:', batchOutput[0][0]);
    console.log('Prediction:');
    console.log('   ', 'token'.padEnd(10), ' ', 'prob'.padEnd(10), ' ');

    // Print the sorted table, marking the target from the batchOutput.
    for (const token of possibleTokenTable) {
      // let tokenId = 0; tokenId < tokenRep.tokens.length; tokenId += 1
      let mark = '';
      if (token.tokenId == singleNextTokenIdxArrayData) {
        mark = ' <- Target';
      }
      console.log('   ', token.str.padEnd(10), ' ', token.prob.toFixed(8), ' ', mark);
    }
  } // infer

  return thisExperimentMetrics;
} // run

function printMetric(setOfExpsName: string, experimentMetrics: [string, Metric[]][],
  metricName: keyof Metric, layerIndex: number = 0) {
  // Will print a new image called {exp_name}_{metric_name}.
  function ifArrayExtractIndex(value: number | Array<number>, index: number): number {
    if (isNumber(value)) {
      return value;
    }
    return (value as Array<number>)[index];
  }

  function filterOutOtherMetrics(value: Metric): {
    [key: string]: number,
    "step": number
  } {
    return {
      [metricName]: ifArrayExtractIndex(value[metricName] ?? 0, layerIndex),
      "step": value["step"],
    }
  }

  let b = experimentMetrics.flatMap(([name, values]) => values.filter(
    (value) => value[metricName] !== undefined).flatMap(d => ({ name, ...filterOutOtherMetrics(d) })));

  let suffix = "";
  let name_suffix = "_layer_" + layerIndex;
  if (metricName == "alphaFirstParams" || metricName == "alphaSecondParams") {
    suffix = " for layer " + layerIndex;
  }

  // Plotting function etc...
  // Construct the correct data array
  let plot = Plot.plot({
    document: new JSDOM("").window.document,
    marks: [
      Plot.frame(),
      Plot.lineY(b, { x: "step", y: metricName, stroke: "name" }),
      Plot.text(b, Plot.selectLast(
        { x: "step", y: metricName, z: "name", text: "name", textAnchor: "start", dx: 3 })),
      Plot.text(['Plot of ' + metricName + ' against number of steps' + suffix], { frameAnchor: "top", dy: -30 }),
    ],
    figure: false,
    x: {
      label: "Step",
      labelAnchor: "center",
    },
    y: {
      label: metricName,
      labelAnchor: "center",
    },
    marginTop: 40,
    marginLeft: 40,
    marginRight: 40,
  });

  const outerHTMLWithBackground: string = plot.outerHTML.replace(
    "<style>",
    `<rect width="100%" height="100%" fill="white"/><style>`
  );

  async function callSharp(): Promise<void> {
    try {
      const buffer: Buffer = await sharp(Buffer.from(outerHTMLWithBackground, "utf-8")).png().toBuffer();
      fs.mkdirSync(setOfExpsName, { recursive: true });
      const nameToSave = setOfExpsName + "/" + metricName + name_suffix + ".png";
      fs.writeFileSync(nameToSave, buffer);
      console.log("PNG image saved as " + nameToSave);
    } catch (error) {
      console.error("Error saving PNG:", error);
    }
  }

  callSharp();
}

function launchExperimentsAndPlot(setOfExpsName: string, partialConfigs: Partial<ExperimentConfig>[]) {
  // First set other arguments:
  const configs = partialConfigs.map((value) => defaultConfigs(value));
  console.log("Launching experiment " + setOfExpsName);
  console.log("Number of experiments is " + configs.length);
  console.log("Configs are:");
  console.log(configs);
  let experimentMetrics: [string, Metric[]][] = [];
  configs.map((config) => experimentMetrics.push([config.name, run(config)]));

  // Plot metrics.
  for (const metric of availableMetrics) {
    if (metric == "loss" || metric == "accuracy")
      printMetric(setOfExpsName, experimentMetrics, metric as keyof Metric);
    else {
      for (let i = 0; i < configs[0].nHeads; i++) {
        printMetric(setOfExpsName, experimentMetrics, metric as keyof Metric, i);
      }
    }
  }
}

// const cfgs: ExperimentConfig[] = [{
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "dynamic_residuals",
//   "seed": 42,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "with_residuals",
//   "seed": 42,
//   "useAlphaParams": false,
//   "useResiduals": true,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "no_residuals",
//   "seed": 42,
//   "useAlphaParams": false,
//   "useResiduals": false,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// }]
// const cfgs: ExperimentConfig[] = [{
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "seed 1",
//   "seed": 1,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "seed 2",
//   "seed": 2,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 3,
//   "name": "seed 3",
//   "seed": 3,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 3,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// }]
// const cfgs: ExperimentConfig[] = [{
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 6,
//   "name": "seed 1",
//   "seed": 1,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 6,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 6,
//   "name": "seed 2",
//   "seed": 2,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 6,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// },
// {
//   "learningRate": 1e-3,
//   "nBatchSize": 64,
//   "nHeads": 6,
//   "name": "seed 3",
//   "seed": 3,
//   "useAlphaParams": true,
//   "useResiduals": false,
//   "startFreezingAtIndex": 6,
//   "unfreezeEveryNSteps": 1000,
//   "nIterations": 300,
// }]

const cfgs: Partial<ExperimentConfig>[] = [{
  "name": "w alpha",
  // 0 becomes negative
  "seed": 42,
  "useAlphaParams": true,
  "useResiduals": false,
  "nIterations": 5,
},
{
  "name": "w residuals",
  "seed": 42,
  "useAlphaParams": false,
  "useResiduals": true,
  "nIterations": 5,
},
{
  "name": "no residuals",
  "seed": 42,
  "useAlphaParams": false,
  "useResiduals": false,
  "nIterations": 5,
}]

// TODO(@aliciafmachado): we should dump the metrics and configs somewhere with the identifier.
// TODO(@aliciafmachado): we can perhaps add an additional identifier twith a timestamp so that the name is unique.
// TODO(@aliciafmachado): we need to plot accuracy every K steps. And to compute what's the best accuracy and loss.
// TODO(@aliciafmachado): we want to run a few experiments and compile it in a doc.

launchExperimentsAndPlot("test", cfgs);
