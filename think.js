var
fs = require('fs'),
NeuralNetwork = require('./NeuralNetwork'),
DataIris = require('./DataIris'),
inputs = [],
targets = []

DataIris.read('data/bezdekIris.data', inputs, targets)

var nn = new NeuralNetwork(JSON.parse(fs.readFileSync('./memory',{encoding:'utf8'})))
console.log(targets[0], nn.think(inputs[0]))
