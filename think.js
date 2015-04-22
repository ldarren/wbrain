var
fs = require('fs'),
NeuralNetwork = require('./NeuralNetwork'),
DataIris = require('./DataIris'),
set = process.argv[2] || 0,
inputs = [],
targets = []

DataIris.read('data/test.data', inputs, targets)

if (set >= inputs.length) set = inputs.length-1

var nn = new NeuralNetwork(JSON.parse(fs.readFileSync('./memory',{encoding:'utf8'})))
console.log(targets[set], nn.think(inputs[set]))
