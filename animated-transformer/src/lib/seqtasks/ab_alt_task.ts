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
Simple generative task for stings of the form `a?(ba)*b?`
(If followed, `a` can only be followed by `b` and b can only be followed by `a`)

// Ideas for other tasks: copy string.
//   ==>  LSTM + one attention layer makes it learn quickly.
*/

import { RandomState, RandomStream, makeRandomStream } from '../random/random';
import { StateIter } from '../state-iter/state-iter';
import { BasicLmTask, RandLmTaskConfig, Example, BasicRandLmTask } from './util';

export const baseVocab = ['a', 'b'];

export type AorBisMaxTaskConfig = RandLmTaskConfig & {
  kind: 'AorBisMaxTask';
};

export class AorBisMaxTask implements BasicRandLmTask {
  public name = 'AorBisMaxTask';
  public baseVocab = baseVocab;
  private exampleId = 0;
  public exampleIter: StateIter<RandomState, Example>;

  constructor(public config: AorBisMaxTaskConfig) {
    this.exampleIter = new StateIter(structuredClone(this.config.genStateConfig), (r) =>
      this.examplesGen(r)
    );
  }

  // Problem Descriptions:
  // * What's the pair of number's with the biggest difference, that you can swap
  // to improve the ordering of the list.
  // * Of all pairs that you can swap to improve the ascending ordering of the list,
  // what pair have the biggest difference?
  genRandExample(r: RandomState): Example {
    const rng = new RandomStream(r);
    const input = new Array<string>(this.config.maxInputLen);
    const output = new Array<string>(this.config.maxOutputLen);
    input[0] = rng.randomEntryFromList(['a', 'b']);
    for (let i = 1; i < input.length; i++) {
      input[i] = input[i - 1] === 'a' ? 'b' : 'a';
    }
    for (let i = 0; i < output.length; i++) {
      output[i] = output[i - 1] === 'a' ? 'b' : 'a';
    }
    return { id: this.exampleId++, input, output };
  }

  *examplesGen(rng: RandomState): Generator<Example, undefined, undefined> {
    while (true) {
      yield this.genRandExample(rng);
    }
  }
}

// Issue: need to make sure that input size is consistent.
// CONSIDER: maybe tasks should be purely string => string?
//
// export function* dataGenerator(
//   config: AbTaskConfig,
//   inputEmb: TokenEmb, outputEmb: TokenEmb, numElements: number
// ): Iterator<tf.TensorContainerObject> {
//   let index = 0;
//   while (index < numElements) {
//     index++;
//     const example = genRandExample(config);
//     yield {
//       xs: inputEmb.embed1(example.input),  // inputs
//       ys: outputEmb.embed1(example.output)  // correct outputs
//     } as tf.TensorContainerObject;
//   }
// }
