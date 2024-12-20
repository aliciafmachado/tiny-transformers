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

import { boolean, number, string } from 'yargs';
import { DictArrTree } from '../js_tree/js_tree';

// Determistic stateful iterables. The assumption is that the state contains
// all the information used by the iterator, and the iterator updates the state.
// Copying the state, effectively enables

export interface Copyable<S> {
  copy(): S;
}

export type UnknownCopyable = {
  copy(): UnknownCopyable;
};

export class CopyableData<T> implements Copyable<CopyableData<T>> {
  constructor(public data: T) {}
  copy(): CopyableData<T> {
    return new CopyableData(structuredClone(this.data));
  }
}

// Class wrapper so we have convenience functions handy.
export class StateIter<S extends Copyable<S>, T>
  implements Iterable<T>, Iterator<T>, Copyable<StateIter<S, T>>
{
  constructor(public state: S, public iterFn: (s: S) => Iterator<T>) {
    this.state;
  }

  copy(): StateIter<S, T> {
    return new StateIter(this.state.copy(), this.iterFn);
  }

  filter(filterKeepFn: (i: T) => boolean): void {
    const originalIterFn = this.iterFn;
    const newIterFn = (state: S) =>
      filterGen(filterKeepFn, originalIterFn(state));
    this.iterFn = newIterFn;
  }

  map<T2>(fn: (x: T) => T2): StateIter<S, T2> {
    const newState = structuredClone(this.state);
    const oldIterFn = this.iterFn;
    function newIterFn(s: S): Iterator<T2> {
      return mapGen(oldIterFn(s), fn);
    }
    return new StateIter(newState, newIterFn);
  }

  // Returns a state iterator copy for the first N examples.
  takeOutN(n: number): T[] {
    return listNextN(this.iterFn(this.state), n);
  }

  next() {
    return this.iterFn(this.state).next();
  }

  [Symbol.iterator]() {
    return this.iterFn(this.state);
  }
}

// export function generateBatch<T>(
//   exampleGenFn: (
//     globalExampleId: number,
//     batchId: number,
//     localExampleId: number) => T,
//   config: {
//     batchSize: number,
//     globalExampleId: number,
//     batchId: number
//   }) {
//   const examples: T[] = [];
//   for (let localExampleId = 0; localExampleId < config.batchSize; localExampleId++) {
//     examples.push(
//       exampleGenFn(config.globalExampleId, config.batchId, localExampleId));
//   }
//   return examples;
// }

// export function* generateBatches<T>(
//   exampleGenFn: (
//     globalExampleId: number,
//     batchId: number,
//     localExampleId: number) => T,
//   config: {
//     batchSize: number,
//     nBatches?: number
//   }
// ): Generator<{ batchId: number, examples: T[] }, undefined, undefined> {
//   let globalExampleId = 0;
//   for (let batchId = 0; !config.nBatches || batchId < config.nBatches; batchId++) {
//     const examples = generateBatch(exampleGenFn,
//       { batchSize: config.batchSize, globalExampleId, batchId });
//     globalExampleId++;
//     yield { batchId, examples };
//   }
//   return;
// }

// export function* generateTaskData(
//   task: BasicLmTask,
//   tokenEmb: TokenEmb,
//   numElements?: number
// ): Iterator<tf.TensorContainerObject> {
//   let index = 0;
//   const examplesGen = task.makeExamplesGenerator();
//   while (numElements && index < numElements) {
//     index++;
//     const example = examplesGen.next();
//     if (example.done) {
//       return;
//     }
//     yield {
//       xs: embed(example.value.input).tensor, // inputs
//       ys: embed(example.value.output).tensor, // correct outputs
//     } as tf.TensorContainerObject;
//   }
// }

// Maybe remove, it is special case of the below?
export function* listGen<T>(l: T[]): Iterator<T> {
  for (const i of l) {
    yield i;
  }
  return;
}

// export function splitStatefulIterAtFirstN<T>(
//   iter: DIterable<T>,
//   n: number
// ): IterSplit<T> {
//   const g = iterToGen(iter);
//   const first = takeNextNgen(g, n);
//   const iterSplit: DIterSplit<T> = {
//     first,
//     rest: g as DIterable<T>,
//   };
//   return iterSplit;
// }

export function listNextN<T>(iter: Iterator<T>, n: number): T[] {
  if (n <= 0) {
    return [] as T[];
  }
  const l: T[] = [];
  let r: IteratorResult<T, unknown> = iter.next();
  n--;
  while (!r.done) {
    l.push(r.value);
    if (n-- <= 0) {
      break;
    }
    r = iter.next();
  }
  return l;
}

export function* takeNextN<T>(iter: Iterable<T>, n: number): Iterable<T> {
  for (const i of iter) {
    if (n-- <= 0) {
      break;
    }
    yield i;
  }
  return;
}

export function* mapGen<T, R, T2>(
  g: Iterator<T, R, undefined>,
  fn: (x: T) => T2
): Generator<T2, R | undefined, undefined> {
  while (true) {
    // argument to g.next is undefined (would be T3)
    const curVal: IteratorResult<T, R> = g.next();
    if (curVal.done) {
      return curVal.value; // type is T2
    }
    yield fn(curVal.value); // type is T1
  }
}

export function* takeNextNgen<T, T2>(
  g: Iterator<T, T2, undefined>,
  n: number
): Generator<T, T2 | undefined, undefined> {
  while (n-- > 0) {
    // argument to g.next is undefined (would be T3)
    const curVal: IteratorResult<T, T2> = g.next();
    if (curVal.done) {
      return curVal.value; // type is T2
    }
    yield curVal.value; // type is T1
  }
  return;
}

export function* filterIterable<T>(
  filterFn: (x: T) => boolean,
  iter: Iterable<T>
): Iterable<T> {
  for (const i of iter) {
    if (filterFn(i)) {
      yield i;
    }
  }
  return;
}

export function* filterGen<T, T2>(
  filterFn: (x: T) => boolean,
  g: Iterator<T, T2, undefined>
): Generator<T, T2, undefined> {
  let curVal: IteratorResult<T, T2>;
  curVal = g.next();
  while (!curVal.done) {
    if (filterFn(curVal.value)) {
      yield curVal.value;
    }
    curVal = g.next();
  }
  return curVal.value;
}

// export function* filterItor<T>(
//   filterKeepFn: (x: T) => boolean,
//   g: Iterator<T>
// ): Iterator<T> {
//   let curVal: IteratorResult<T>;
//   curVal = g.next();
//   while (!curVal.done) {
//     if (filterKeepFn(curVal.value)) {
//       yield curVal.value;
//     }
//     curVal = g.next();
//   }
//   return curVal.value;
// }
