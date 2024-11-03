# Downloading Gemma

You will find instructions here on how to download and convert Gemma 2 to TF JS.

First of all, you need to download the model in the **Keras** format [here](https://www.kaggle.com/models/google/gemma/keras/gemma_2b_en?postConsentAction=download). Unfortunately, this needs to be done manually as you will need to accept the conditions for downloading Gemma.

Once you have your model, you should paste your `.h5` file under a new `pretrained_models` directory under `tiny-transformers`.

Then you have two options:

1. Use python to convert the model or;
2. Run a command in your terminal to convert the model.

Both instructions are available in TF JS documentation. Here we will show how to do it running a command in your terminal. You may want to create an isolated environment using Python 3.11.

First of all, you will need create a Python environment with Python 3.11 and install the required packages.

Run the command below under the `tiny-transformers` directory:

Make sure to define `$MODEL_PATH` to be your h5 file, e.g. `/path/to/model.h5`.

```bash
mkdir -p pretrained_js_models $MODEL_PATH &&
tensorflowjs_converter --input_format keras \
                       $MODEL_PATH \
                       pretrained_js_transformers
```
