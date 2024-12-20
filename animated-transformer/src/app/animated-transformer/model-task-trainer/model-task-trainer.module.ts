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


import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModelTaskTrainerComponent } from './model-task-trainer.component';

import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { CodemirrorConfigEditorModule } from '../../codemirror-config-editor/codemirror-config-editor.module';
import { D3LineChartModule } from '../../d3-line-chart/d3-line-chart.module';
import { AutoCompletedTextInputComponent } from '../../auto-completed-text-input/auto-completed-text-input.component';

@NgModule({
  declarations: [
    ModelTaskTrainerComponent
  ],
  imports: [
    CommonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    FormsModule,
    ReactiveFormsModule,
    MatSlideToggleModule,
    MatListModule,
    MatMenuModule,
    MatAutocompleteModule,
    CodemirrorConfigEditorModule,
    D3LineChartModule,
    AutoCompletedTextInputComponent,
  ],
  exports: [
    ModelTaskTrainerComponent
  ]
})
export class ModelTaskTrainerModule { }
