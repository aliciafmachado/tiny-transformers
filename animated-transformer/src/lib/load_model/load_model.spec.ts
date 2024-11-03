// TODO(@aliciafmachado): Figure out how to test loading the model.

import * as tf from '@tensorflow/tfjs-node';


describe('stringify', () => {
    it('basic stringifyJsonVaue', async () => {
        console.log("hello world")
        // const handler = tf.io.fileSystem("/Users/afm/github/tiny-transformers/pretrained_js_models/gemma-2-2b/model.json");
        const model = await tf.loadLayersModel('file://Users/afm/github/tiny-transformers/pretrained_js_models/gemma-2-2b/model.json'); // 'file:///Users/afm/github/tiny-transformers/pretrained_js_models/gemma-2-2b/model.json'
        

    expect(0)
    .toEqual(0);
    });
})