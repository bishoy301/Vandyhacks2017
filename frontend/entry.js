import { Model } from 'keras-js'

const model = new Model({
  filepaths: {
    model:  'model.json',
    weights: 'model_weights.buf',
    metadata: 'model_metadata.json'
  },
  gpu: true
})