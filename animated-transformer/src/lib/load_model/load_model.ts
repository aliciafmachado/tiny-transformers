import * as tf from '@tensorflow/tfjs-node';

export async function loadModel(modelPath: string) {
    console.log("Loading model from ", modelPath);
    const model = await tf.loadLayersModel(modelPath);
    return model;
}