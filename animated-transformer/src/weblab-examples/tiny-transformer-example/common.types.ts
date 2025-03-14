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

import { TransformerConfig, TransformerParams } from 'src/lib/transformer/transformer_gtensor';
import { Metrics } from 'src/lib/distr-signals/cell-kind';
import { SerializeTensorParams } from 'src/lib/gtensor/params';

export type SimpleMetrics = Metrics<'entropyLoss' | 'accuracy'>;

export type Batch = {
  batchId: number; // just a counter
  nextSeed: number; // Unique ID that generates the batch.
  inputs: string[][]; // every example input is a string[] of tokens.
  outputs: string[][]; // every example output is a string[] of tokens.
};

export type TrainConfig = {
  id: string;
  kind: 'basicSeqTrainer';
  // Training hyper-params
  randomSeed: number;
  learningRate: number;
  batchSize: number;
  maxInputLength: number;
  trainForBatches: number;
  // Eval/reporting/saving
  checkpointFrequencyInBatches: number;
  metricReporting: {
    metricFrequencyInBatches: number;
  };
};

export enum ModelInitKind {
  ReinitFromConfig = 'ReinitFromConfig',
  ReplaceParams = 'ReplaceParams',
  ReplaceParamsAndConfig = 'ReplaceParamsAndConfig',
  Null = 'Null',
}

export type ModelInit =
  | {
      kind: ModelInitKind.ReplaceParams;
      config: TransformerConfig;
      serializedParams: SerializeTensorParams<TransformerParams>;
    }
  | {
      kind: ModelInitKind.ReinitFromConfig;
      config: TransformerConfig;
    }
  | {
      kind: ModelInitKind.ReplaceParamsAndConfig;
      config: TransformerConfig;
      serializedParams: SerializeTensorParams<TransformerParams>;
    };

export type EnvModel = {
  config: TransformerConfig;
  serializedParams?: SerializeTensorParams<TransformerParams>;
};

export type Checkpoint = {
  config: TransformerConfig;
  serializedParams: SerializeTensorParams<TransformerParams>;
  lastBatch: Batch;
  metrics: SimpleMetrics;
};

export type TaskGenConfig = {
  initBatchId: number;
  initBatchSeed: number;
  testSetSize: number;
  maxBatches: number;
  batchSize: number;
};
