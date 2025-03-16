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

import {
  TransformerComputation,
  lastTokenLogits,
  allPastTokensCrossEntropyLossWithIntegerLabels,
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

const tfjsBackendName = tf.getBackend();
console.log('tfjs backend:', tfjsBackendName);

const printEveryNBatches = 10;
const useResiduals = true;
const useAlphaParams = false;
const learningRate = 1e-3;
const nIterations = 300;
const nBatchSize = 64;
const unfreezeEveryNSteps = 50;


function getTaskConfig(): TinyWorldTaskConfig {
  const taskConfig: TinyWorldTaskConfig = {
    ...defaultTinyWorldTaskConfig,
    maxInputLen: 10,
    maxOutputLen: 1,
  };
  return taskConfig;
}

function initTransformerConfig(baseVocab: string[], nHeads: number = 6, alphaParams: boolean = false,
  residuals: boolean = true,
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
      seed: 42,
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
  for (let batchId = 0; batchId < batchNum; batchId += 1) {
    let batchOriginal = task.exampleIter.takeOutN(batchSize);
    let inputs = batchOriginal.map((example) => example.input);
    let outputs = batchOriginal.map((example) => example.output);
    yield { batchId, inputs, outputs };
  }
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
  batchOutput: string[][]
): tf.Scalar {
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
  const entropyLoss: tf.Scalar = allPastTokensCrossEntropyLossWithIntegerLabels(model, computation, targetTokens);
  if (batchId % printEveryNBatches === 0) {
    console.log(
      `batch: ${batchId} `.padEnd(15) +
      ('entropyLoss: ' + entropyLoss.arraySync().toFixed(8)).padEnd(25)
    );
    console.log(
      `alphaFirst: ${model.params.layers.map((g) => g.alphaParams?.alphaFirst.tensor.asScalar().arraySync().toFixed(8))}`
    )
    console.log(
      `alphaSecond: ${model.params.layers.map((g) => g.alphaParams?.alphaSecond.tensor.asScalar().arraySync().toFixed(8))}`
    )
    // TODO: print grads

  }
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

// function testParams(transformerParams: jstree.DictArrTree<GTensor<any>>): jstree.DictArrTree<GTensor<any>> {
//   return transformerParams;
// }

function run() {
  // define task
  const trainTaskConfig = getTaskConfig();
  const trainTask = new TinyWorldTask(trainTaskConfig);

  // define vocab & decoder
  const Config = initTransformerConfig(trainTask.baseVocab, 6, useAlphaParams, useResiduals);
  const decoderParams = varifyParams(initDecoderParams(Config));
  const model: TransformerModel = {
    config: Config,
    params: decoderParams as TransformerParams,
  };
  const randomStream = makeRandomStream(42);

  // By manipulating decoderParams, you can selectively limit what parameters
  // get tuned. By manipulating, we mean changing the trainable state to false.
  initParametersTrainableButAlphaFrom(decoderParams as TransformerParams, 1);
  let paramsList = listifyVarParams(decoderParams).map((g) => g.variable);
  let unfreezeId = 1;

  {
    // train with optimization
    const batchNum: number = nIterations;
    const batchSize: number = nBatchSize;

    let optimizer = tf.train.adam(learningRate);
    for (let batch of batchGenerator(trainTask, batchNum, batchSize)) {
      let { batchId, inputs, outputs } = batch;
      optimizer.minimize(
        () => computeLoss(model, randomStream, batchId, inputs, outputs),
        false,
        paramsList
      );
      batchId += 1;

      if (batchId % unfreezeEveryNSteps == 0 && unfreezeId < decoderParams.layers.length) {
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
} // run

run();
