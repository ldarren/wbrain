// dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
const
IRIS_CLASS = {'Iris-setosa':[1,0,0],'Iris-versicolor':[0,1,0],'Iris-virginica':[0,0,1]}

var
fs = require('fs'),
path = require('path'),
NeuralNetwork = require('./NeuralNetwork'),
toNum = function(i){ return parseFloat(i, 10) },
readDataSet = function(fpath, inputs, targets){
    inputs.length = 0
    targets.length = 0
    var data = fs.readFileSync(path.isAbsolute(fpath) ? fpath : path.resolve(__dirname+path.sep+fpath), {encoding:'utf8'})
    if (!data) throw 'given path ['+fpath+'] not found!'
	var lines = data.split('\n')
    if (!lines || !lines.length) throw 'empty dataset?'

    for(var i=0,l=lines.length,arr; i<l; i++){
        arr = lines[i].split(',')
        if (!arr || 2>arr.length) break
        targets.push(IRIS_CLASS[arr.splice(-1, 1)[0]])
        inputs.push(arr.map(toNum))
    }
},
inputs = [],
targets = []

readDataSet('data/bezdekIris.data', inputs, targets)

var nn = new NeuralNetwork([inputs[0].length, 7, targets[0].length])
console.log(nn.learn(100000, inputs, targets))
//console.log(nn.think(inputs[0]))
fs.writeFileSync('./memory', JSON.stringify(nn.memory()))

/*
var nn = new NeuralNetwork(JSON.parse(fs.readFileSync('./memory',{encoding:'utf8'})))
//nn.learn(1, inputs, targets)
console.log(nn.think(inputs[0]))
*/
