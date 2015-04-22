var
fs = require('fs'),
NeuralNetwork = require('./NeuralNetwork'),
DataIris = require('./DataIris'),
set = process.argv[2],
inputs = [],
targets = []

DataIris.read('data/bezdekIris.data', inputs, targets)

var nn = new NeuralNetwork(JSON.parse(fs.readFileSync('./memory',{encoding:'utf8'})))
console.log(targets[set], nn.think(inputs[set]))
