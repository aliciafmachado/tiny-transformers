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

import {
  Component,
  Input,
  OnInit,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  signal,
  WritableSignal,
  Signal,
  computed,
} from '@angular/core';

import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatTableModule } from '@angular/material/table';
import { MatCardModule } from '@angular/material/card';
import { AutoCompletedTextInputComponent } from 'src/app/auto-completed-text-input/auto-completed-text-input.component';

import json5 from 'json5';
import * as swap_task from '../../../lib/seqtasks/swap_task';
import {
  DecisionBoundaryTask,
  DecisionBoundaryTaskConfig,
  baseVocab as dboundaryVocab,
} from '../../../lib/seqtasks/decision_boundary_task';
import { stringifyJsonValue } from '../../../lib/json/pretty_json';
import {
  BasicLmTask,
  BasicLmTaskConfig,
  BasicLmTaskUpdate,
  RandLmTaskConfig,
  Example,
  BasicRandLmTask,
} from 'src/lib/seqtasks/util';
import { Output, EventEmitter } from '@angular/core';
import {
  ConfigUpdate,
  ConfigUpdateKind,
} from 'src/app/codemirror-config-editor/codemirror-config-editor.component';
import { SecretTokenTask, SecretTokenTaskConfig } from 'src/lib/seqtasks/secret_token_task';
import {
  TinyWorldTask,
  TinyWorldTaskConfig,
  defaultTinyWorldTaskConfig,
  tinyWorldTaskKind,
} from 'src/lib/seqtasks/tiny_worlds';
import { ConfigKind, ConfigObj } from 'src/lib/json/config-obj';
import { taskRegistry } from 'src/lib/seqtasks/task_registry';
import * as _ from 'underscore';
import { nullableEqFn } from 'src/lib/utils';
// import { TinyModelsService } from 'src/app/tiny-models.service';

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
@Component({
  selector: 'app-seq-task-selector',
  templateUrl: './seq-task-selector.component.html',
  styleUrls: ['./seq-task-selector.component.scss'],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    // ---
    MatButtonModule,
    MatIconModule,
    MatInputModule,
    MatMenuModule,
    MatListModule,
    MatAutocompleteModule,
    MatTableModule,
    MatCardModule,
    AutoCompletedTextInputComponent,
  ],
})
export class SeqTaskSelectorComponent {
  view: 'edit' | 'view' = 'view';

  showExamples = false;
  shownNumOfExamples = 6;
  repSize = 8;

  currentTask = signal<BasicRandLmTask | null>(null, {
    equal: nullableEqFn((a, b) => _.isEqual(a.config, b.config)),
  });
  selectedTaskExamples!: Signal<Example[] | null>;
  datasetColumns: string[] = ['input', 'target'];

  get taskNames(): string[] {
    return [];
    // return Object.keys(this.tmService.taskConfigsMap);
  }

  constructor() {} // public tmService: TinyModelsService

  currentTaskConfigStr(): string {
    return '';
    // return stringifyJsonValue(this.tmService.modelConfig());
  }

  selectTask(maybeName: string | null): void {
    // this.tmService.selectTask(maybeName);
  }

  taskConfigUpdated(configUpdate: ConfigUpdate<RandLmTaskConfig>): void {
    // When configUpdate has a new object, we assume it to be correct.
    //
    // TODO: provide some runtime value type checking. Right now all that is
    // needed is valid JSON/JSON5, but if you provide valid JSON missing needed
    // values (e.g. encoderConfig is null), it should complain here, but
    // currently does not.
    if (configUpdate.close) {
      this.view = 'view';
    }
    if (configUpdate.kind !== ConfigUpdateKind.UpdatedValue) {
      return;
    }
    // this.tmService.updateTaskConfig(configUpdate.obj);
  }

  toggleEditor() {
    this.view = this.view === 'edit' ? 'view' : 'edit';
  }

  inputToString(input: string[]): string {
    return JSON.stringify(input);
  }
  outputToString(output: string[]): string {
    return JSON.stringify(output);
  }

  taskConfigAsJson(config: RandLmTaskConfig): string {
    return stringifyJsonValue(config, {
      arrWrapAt: 60,
      objWrapAt: 60,
      curIndent: '',
      sortObjKeys: true,
    });
    // json5.stringify(config, null, 2);
  }
}
