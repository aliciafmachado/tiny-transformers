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

/* Implements tests for GPT2 in TS */

/* Check number of parameters */

import { GTensor, makeTruncNormal } from '../gtensor/gtensor';
import * as transformer from './transformer_gtensor';
import { AttnHeadParamSpec, AttnHeadComputeSpec, TransformerConfig, TransformerParamLayerSpec } from './gpt2';
import * as tf from '@tensorflow/tfjs';
import * as abtask from '../seqtasks/ab_task';
import { BasicTaskTokenRep, embedBatch, prepareBasicTaskTokenRep } from '../tokens/token_gemb';
import { makeRandomStream } from '../state-iter/random';
import* as jstree from '../js_tree/js_tree';

describe('GTensor Transformers', () => {
  it('basic transformer shapes', () => {
    // const paramSizes: AttnHeadParamSpec = {
    //     inputRep: 1024,
    //     hiddenRep: 784,
    //     kq: 784, // same nb as value i think
    //     heads: 12,
    //     value: 784,
    //     layerNormHeadsProjection: true, // need to change this to follow the implementation
    //     layerNormFF: true,
    //     addLayerNormBias: true,
    // };
    
    // 12 Heads.
    const transformer_param_layer_spec: TransformerParamLayerSpec = {
        nHeads: 12,
        // need to pass 1024 somehow to hasPosEncoding
        hasPosEncoding: true,
        computeSpec: { residuals: true, dropoutRate: 0.0 },
        layerNormFF: true,
        layerNormHeadsProjection: true,
        addLayerNormBias: true,
      };

    // To be checked if this is right
    // 12 Layers.
    const gpt2: TransformerConfig = {
          spec: {
            inputRep: 768,
            hiddenRep: 768,
            kqvRep: 768,
            layers: Array(12).fill(transformer_param_layer_spec),
            dropoutRate: 0.0,
            // This below is not doing anything: need to check what's happening.
            relPosEncodingSeqLength: 1024,
            // Missing a layer norm outside.
          },
          init: {
            stddev: 0.5,
            mean: 0,
            seed: 96,
          },
      };
    // progress: embeddings is correct
    // heads have more parameters than expected
    // and i don't think that the positional encoders are in the parameters of the model
    // so that's why it's not impacting the final number of parameters.
    
    const tokens = Array(50257).fill("test");
    // The BasicTaskTokenRep below is not valid but it's fine since we are just checking the
    // number of parameters.
    const tokenRep: BasicTaskTokenRep = {
        maskToken: "test", 
        padToken: "test", 
        eosToken: "test", 
        tokens: tokens, 
        tokenToIdx: {},
    };
    const params = transformer.initDecoderParamsTree(
        tokenRep, 
        gpt2);
    // const params = transformer.initAttnHeadParams(paramSizes);
    const paramCount = jstree.reduce<GTensor<any>, number>(
        (count, paramObj) => count + paramObj.tensor.size,
        0,
        params.tokenEmbedding
      );
      console.log(paramCount);
      // Test 2: Check number of parameters in GPT2.
      expect(paramCount).toEqual(124439808);
      
      // Check if the output matches the one from the implementation in python.
      // TODO (@aliciafmachado)
    // const inputExample1 = new GTensor(
    //   tf.tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //       [5, 6],
    //     ],
    //   ]),
    //   ['batch', 'pos', 'inputRep']
    // );
    // const generator = makeRandomStream(0);
    // const parts = transformer.computeAttnHead(spec, params, inputExample1, generator);
    // expect(parts.attendedValues.dimNames).toEqual(
    //   jasmine.arrayContaining(['batch', 'heads', 'value', 'pos'])
    // );
    // expect(parts.attendedValues.gshape()).toEqual({
    //   batch: 1,
    //   heads: 1,
    //   value: 4,
    //   pos: 3,
    // });
  });

//   it('AB task data prep', async () => {
//     const inputRep = 2;
//     const batchSize = 4;
//     const task = new abtask.AorBisMaxTask({
//       name: 'AorBisMaxTask',
//       maxInputLen: 2,
//       maxOutputLen: 2,
//       seed: 0,
//       // Create a tokenEmbedding that also has [MASC] token & [PAD] token.
//       // inputRepSize: inputRep,
//     });
//     const tokenRep = prepareBasicTaskTokenRep(task.baseVocab);
//     const padTokenId = tokenRep.tokenToIdx[tokenRep.padToken];
//     const embeddings = makeTruncNormal({
//       tokenId: tokenRep.tokens.length,
//       inputRep,
//     });

//     // len = taskConfig.batchSize
//     const examples = task.exampleIter.takeOutN(4);

//     const batchedInputEmb = embedBatch(
//       tokenRep.tokenToIdx,
//       embeddings,
//       examples.map((example) => example.input.concat(tokenRep.maskToken)),
//       { paddingId: padTokenId, padAt: 'start', dtype: 'int32' }
//     );

//     expect(batchedInputEmb.gshape()).toEqual({
//       batch: batchSize,
//       // +1 for the appended [MASK] token to be predicted.
//       pos: task.config.maxInputLen + 1,
//       inputRep,
//     });
//   });
});