import { Component, Input, Output, EventEmitter, signal } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Router, ActivatedRoute } from '@angular/router';
import { FormControl } from '@angular/forms';
// TODO(@aliciafmachado): figure out how to import the model.
// import { GPT2 } from '@tensorflow/tensorflow-models/gpt2/src/gpt2';
// import { load } from '@tfjs-models/gpt2/gpt2';

@Component({
  selector: 'app-load-model',
  templateUrl: './load-model.component.html',
  styleUrl: './load-model.component.scss'
})
export class LoadModelComponent {
  path_to_model = signal('' as string);

  // @Input()
  // set inputValue(inputUpdateStr: string | null) {
  //   this.inputControl.setValue(inputUpdateStr);
  // }

  @Output() evalInputUpdate = new EventEmitter<string>();

  constructor() {
    // this.inputControl = new FormControl<string | null>('', this.path_to_model);
    // this.inputControl.valueChanges.forEach((s) => {
    //   this.setInputValueFromString(s);
    //   if (s !== null) {
    //     this.evalInputUpdate.emit(s);
    //   }
    // });
  }

  // setInputValueFromString(s: string | null) {
  //   if (s !== null && !jsonStrListErrorFn(this.validatorConfig, s)) {
  //     this.input.set(json5.parse(s));
  //   }
  // }

  updatePathToModel(path: string) {
    this.path_to_model.set(path);
  }

  loadModelAndPrintWeights() {
    // Will do something with the path to the model.
    // Unfortunately, I can't import the repository installed through github.
    // Not sure why...
    // load().then((model: GPT2) => {
    //   console.log(model);
    // });
  }
}
