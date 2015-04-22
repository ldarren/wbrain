var
fs = require('fs'),
NeuralNetwork = require('./NeuralNetwork'),
DataIris = require('./DataIris'),
inputs = [],
targets = [],
epochs = process.argv[2] || 2000,
eta = process.argv[3] || 0.05,
alpha = process.argv[4] || 0.01

DataIris.read('data/train.data', inputs, targets)

var nn = new NeuralNetwork([inputs[0].length, 7, targets[0].length], {eta:eta,alpha:alpha,smallwt:0.0005})
nn.learn(epochs, inputs, targets)
fs.writeFileSync('./memory', JSON.stringify(nn.memory()))
