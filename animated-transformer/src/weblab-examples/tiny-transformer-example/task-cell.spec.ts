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
import { SignalSpace } from 'src/lib/signalspace/signalspace';
import { Batch, TaskGenConfig } from './common.types';
import { taskCellKind } from './task-cell.kind';
import { LabEnv } from 'src/lib/distr-signals/lab-env';
import { defaultTinyWorldTaskConfig } from 'src/lib/seqtasks/tiny_worlds';

describe('tiny-transformer-example/task-cell', () => {
  beforeEach(() => {});

  it('simple task cell test: make 5 batches of data', async () => {
    const space = new SignalSpace();
    const env = new LabEnv(space);
    const { setable } = space;

    const taskConfig = setable(defaultTinyWorldTaskConfig);
    const genConfig = setable<TaskGenConfig>({
      initBatchId: 0,
      initBatchSeed: 0,
      maxBatches: 5,
      batchSize: 10,
      testSetSize: 3,
    });
    const task = env.start(
      taskCellKind,
      new Worker(new URL('./task-cell.worker', import.meta.url)),
      { inputs: { taskConfig, genConfig } },
    );
    const testSet = await task.cell.outputs.testSet.connect();
    expect(testSet().length).toEqual(3);

    const trainBatches: Batch[] = [];
    for await (const trainBatch of task.cell.outStreams.trainBatches.connect()) {
      trainBatches.push(trainBatch);
    }
    await task.cell.requestStop();
    await task.cell.onceFinished;

    expect(trainBatches.length).toEqual(5);
    expect(trainBatches[0].batchId).toEqual(0);
    expect(trainBatches[0].inputs.length).toEqual(10);
    expect(trainBatches[0].outputs.length).toEqual(10);
    expect(trainBatches[trainBatches.length - 1].batchId).toEqual(4);
  });
});
