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
import { SignalSpace } from './signalspace';
import { ComputedSValue, WritableSValue } from './savable-value';
import { GTensor } from '../gtensor/gtensor';
import {
  initDecoderParams,
  savableTransformerModelKind,
  transformerModelKind,
} from '../transformer/transformer_gtensor';
import _ from 'underscore';
import { serializeParams } from '../gtensor/params';
import { EnvModel } from 'src/app/web-colab/tiny-transformer-example/ailab';

// type MaybeReturn<T, F> = undefined extends F ? void : T;
type MaybeReturn<T> = void extends T ? void : T;

fdescribe('savable-value', () => {
  async function waitTick<T>(f?: () => T): Promise<MaybeReturn<T>> {
    return new Promise<T | void>((resolve) => {
      setTimeout(() => {
        if (f) {
          resolve(f());
        } else {
          resolve();
        }
      }, 0);
    }) as Promise<MaybeReturn<T>>;
  }

  it('Simple signal compute', () => {
    const s = new SignalSpace();
    const { setable, derived } = s;

    const model = setable<EnvModel>(() => {
      function eqCheck(x: EnvModel, y: EnvModel) {
        
      },

      return {
        config: transformerModelKind.defaultConfig,
        params: initDecoderParams(transformerModelKind.defaultConfig),
      };
    });
    
    // When config changes, update the params.
    derived(() => {
      if (!_.isEqual(model.lastValue().config, config())) {
        params.set(initDecoderParams(transformerModelKind.defaultConfig));
      }
    }

    const modelV = new ComputedSValue(savableTransformerModelKind, model);

    config.update((c) => {
      c.spec.inputRep += 10;
      return c;
    });

    const expectedModifiedConfig = structuredClone(transformerModelKind.defaultConfig);
    expectedModifiedConfig.spec.inputRep += 10;

    expect(modelV.value().config).toEqual(transformerModelKind.defaultConfig);
    expect(modelV.proposedValue().config).toEqual(expectedModifiedConfig);
    modelV.updateValue();
    expect(modelV.value().config).toEqual(expectedModifiedConfig);
  });
});

*/
