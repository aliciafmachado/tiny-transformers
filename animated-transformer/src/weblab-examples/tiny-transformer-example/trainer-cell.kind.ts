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
/**
 * The shared environment between specification between webworkers and the AI
 * lab environment.
 */

import { Example } from 'src/lib/seqtasks/util';
import { Kind, CellKind } from 'src/lib/distr-signals/cell-kind';
import { Batch, Checkpoint, ModelInit, SimpleMetrics, TrainConfig } from './common.types';

export const trainerCellKind = new CellKind('Trainer cell', {
  inputs: {
    modelInit: Kind<ModelInit>,
    trainConfig: Kind<TrainConfig>,
    testSet: Kind<Example[]>,
  },
  inStreams: {
    trainBatches: Kind<Batch>,
  },
  outStreams: {
    metrics: Kind<SimpleMetrics>,
    checkpoint: Kind<Checkpoint>,
  },
});
