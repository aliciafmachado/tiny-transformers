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

import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class TinyModelsService {}

/* import { Injectable } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ConfigKind } from 'src/lib/json/config-obj';
import { tinyWorldTaskKind } from 'src/lib/seqtasks/tiny_worlds';
import { BasicRandLmTask, RandLmTaskConfig } from 'src/lib/seqtasks/util';
import {
  TransformerConfig,
  TransformerModel,
  transformerModelKind,
} from 'src/lib/transformer/transformer_gtensor';
import { SignalSpace, defined } from 'src/lib/signalspace/signalspace';
import { EnvModel, TrainConfig } from './web-colab/tiny-transformer-example/ailab';
import { stringifyJsonValue } from 'src/lib/json/pretty_json';
import { DerivedSignal, SetableSignal } from 'src/lib/signalspace/signalspace';
import { SetableUpdateKind } from 'src/lib/signalspace/setable-node';

const modelMakerMap = {} as { [kind: string]: ConfigKind<TransformerConfig, TransformerModel> };
const initModelConfigsMap = {} as { [id: string]: TransformerConfig };
{
  const initModelKinds = [transformerModelKind];
  initModelKinds.forEach((k) => (modelMakerMap[k.kind] = k));
  const aConfiguredModel = modelMakerMap[transformerModelKind.kind].makeFn(
    transformerModelKind.defaultConfigStr
  );
  initModelConfigsMap[aConfiguredModel.config.id] = aConfiguredModel.config;
}

const taskMakerMap = {} as { [kind: string]: ConfigKind<RandLmTaskConfig, BasicRandLmTask> };
const initTaskConfigMap = {} as { [id: string]: RandLmTaskConfig };
{
  const initTaskKinds = [tinyWorldTaskKind];
  initTaskKinds.forEach(
    (k) => (taskMakerMap[k.kind] = k as never as ConfigKind<RandLmTaskConfig, BasicRandLmTask>)
  );
  const aConfiguredTask = taskMakerMap[tinyWorldTaskKind.kind].makeFn(
    tinyWorldTaskKind.defaultConfigStr
  );
  initTaskConfigMap[aConfiguredTask.config.id] = aConfiguredTask.config;
}

const basicTrainConfig: TrainConfig = {
  id: 'initial config',
  kind: 'basicSeqTrainer',
  // training hyper-params
  learningRate: 0.5,
  batchSize: 64,
  maxInputLength: 10,
  trainForBatches: 10,
  // Reporting / eval
  checkpointFrequencyInBatches: 100,
  metricReporting: {
    metricFrequencyInBatches: 10,
  },
};
const initTrainerConfigMap = {} as { [id: string]: TrainConfig };
const initTrainerKindConfigStrMap = {} as { [kind: string]: string };
{
  const initTrainerConfigs = [basicTrainConfig];
  initTrainerConfigs.forEach((c) => (initTrainerConfigMap[c.id] = c));
  initTrainerKindConfigStrMap[basicTrainConfig.kind] = stringifyJsonValue(basicTrainConfig);
}

@Injectable({
  providedIn: 'root',
})
export class TinyModelsService {
  space: SignalSpace;

  // Tasks...
  taskConfigsMap: { [id: string]: RandLmTaskConfig } = initTaskConfigMap;
  taskConfig: SetableSignal<RandLmTaskConfig | null>;
  get taskId(): string {
    const config = this.taskConfig();
    return config ? config.id : '';
  }
  get taskConfigDefaultStr(): string {
    if (this.taskId === '') {
      return '<undefined>';
    }
    return taskMakerMap[this.taskId].defaultConfigStr;
  }
  task: DerivedSignal<BasicRandLmTask | null>;

  // Models...
  modelConfigsMap: { [id: string]: TransformerConfig } = initModelConfigsMap;
  modelConfig: SetableSignal<TransformerConfig | null>;
  get modelId(): string {
    const config = this.modelConfig();
    return config ? config.id : '';
  }
  get modelConfigDefaultStr(): string {
    if (this.modelId === '') {
      return '<undefined>';
    }
    return modelMakerMap[this.modelId].defaultConfigStr;
  }
  model: DerivedSignal<EnvModel | null>;

  // Optimisers / Trainers...
  trainerConfigsMap: { [id: string]: TrainConfig } = initTrainerConfigMap;
  trainerConfig: SetableSignal<TrainConfig | null>;
  get trainerId(): string {
    const config = this.trainerConfig();
    return config ? config.id : '';
  }
  get trainerConfigDefaultStr(): string {
    if (this.trainerId === '' || !(this.trainerId in this.trainerConfigsMap)) {
      return '<undefined>';
    }
    return initTrainerKindConfigStrMap[this.trainerConfigsMap[this.trainerId].kind];
  }

  constructor(private route: ActivatedRoute, private router: Router) {
    this.space = new SignalSpace();
    const { derivedNullable, setable } = this.space;
    const taskId = Object.keys(this.taskConfigsMap)[0];
    this.taskConfig = setable<RandLmTaskConfig | null>(this.taskConfigsMap[taskId]);
    this.task = derivedNullable(
      () => {
        const config = defined(this.taskConfig);
        return taskMakerMap[config.kind].makeFn(JSON.stringify(config));
      },
      { definedDeps: [this.taskConfig] }
    );
    const modelId = Object.keys(this.modelConfigsMap)[0];
    this.modelConfig = setable<TransformerConfig | null>(this.modelConfigsMap[modelId]);
    // TODO: maybe store modelConfigStr as the source artefact.

    const trainerId = Object.keys(this.modelConfigsMap)[0];
    this.trainerConfig = setable<TrainConfig | null>(this.trainerConfigsMap[trainerId]);

    this.model = derivedNullable<EnvModel>(
      () => {
        // TODO: init params... load them?
        return { config: defined(this.modelConfig) };
      },
      { definedDeps: [this.modelConfig] }
    );

    this.route.queryParams.subscribe((params) => {
      this.selectModel(params['model'] || '');
      this.selectTask(params['task'] || '');
      // this.trainerName = params['trainer'] || '';
      // this.evalInputStr = params['input'] || '';
    });
  }

  selectTask(taskName: string | null) {
    if (taskName === this.taskId) {
      return;
    }

    if (!taskName || !(taskName in this.taskConfigsMap)) {
      this.taskConfig.set(null);
      return;
    }
    this.taskConfig.set(this.taskConfigsMap[taskName]);

    const queryParams = { task: taskName };
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: queryParams,
      // remove to replace all query params by provided
      queryParamsHandling: 'merge',
    });
  }

  selectModel(modelName: string | null) {
    if (modelName === this.modelId) {
      return;
    }

    if (!modelName || !(modelName in this.modelConfigsMap)) {
      this.modelConfig.set(null);
      return;
    }
    this.modelConfig.set(this.modelConfigsMap[modelName]);

    const queryParams = { model: modelName };
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: queryParams,
      // remove to replace all query params by provided
      queryParamsHandling: 'merge',
    });
  }

  updateTaskConfig(config: RandLmTaskConfig) {
    // CONSIDER: a dict type for all tasks that goes with the registry. When a
    // new task is added the registry has to be updated (all registry stuff is
    // then in one place/file, no implicit registry additions).
    taskMakerMap[config.kind].makeFn(JSON.stringify(config));
  }

  updateModelConfig(config: TransformerConfig) {
    console.log('to implement');
  }

  reInitModelParams() {
    this.modelConfig.set(this.modelConfig(), { updateStrategy: SetableUpdateKind.ForceUpdate });
  }
}

*/
