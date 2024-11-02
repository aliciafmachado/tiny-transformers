import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PatchscopesComponent } from './patchscopes.component';

import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { RouterModule } from '@angular/router';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatTableModule } from '@angular/material/table';
import { MatCardModule } from '@angular/material/card';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';

import { CodemirrorConfigEditorModule } from '../codemirror-config-editor/codemirror-config-editor.module';
// import { VegaChartModule } from '../vega-chart/vega-chart.module';
import { D3LineChartModule } from '../d3-line-chart/d3-line-chart.module';
// import { ModelTaskTrainerComponent } from './model-task-trainer/model-task-trainer.component';
// import { NanValidatorDirective } from '../form-validators/nan-validator.directive';
// import { BoundedFloatValidatorDirective } from '../form-validators/bounded-float-validator.directive';
import { AutoCompletedTextInputComponent } from '../auto-completed-text-input/auto-completed-text-input.component';

import { JsonStrListValidatorDirective } from '../form-validators/json-str-list-validator.directive';
import { TokenSeqDisplayComponent } from '../token-seq-display/token-seq-display.component';

@NgModule({
    declarations: [
      PatchscopesComponent,
    ],
    imports: [
        CommonModule,
        BrowserAnimationsModule,
        FormsModule,
        ReactiveFormsModule,
        RouterModule,
        // --
        MatAutocompleteModule,
        MatButtonModule,
        MatCardModule,
        MatFormFieldModule,
        MatIconModule,
        MatInputModule,
        MatListModule,
        MatMenuModule,
        MatSlideToggleModule,
        MatTableModule,
        // ---
        CodemirrorConfigEditorModule,
        // VegaChartModule,
        D3LineChartModule,
        AutoCompletedTextInputComponent,
        TokenSeqDisplayComponent,
    ],
  })
  export class PatchscopesModule {}