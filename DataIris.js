// dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
const
IRIS_CLASS = {'Iris-setosa':[1,0,0],'Iris-versicolor':[0,1,0],'Iris-virginica':[0,0,1]}

var
fs = require('fs'),
path = require('path'),
toNum = function(i){ return parseFloat(i, 10) }

exports.read = function(fpath, inputs, targets){
    inputs.length = 0
    targets.length = 0
    var data = fs.readFileSync(/*path.isAbsolute(fpath)*/0 ? fpath : path.resolve(__dirname+path.sep+fpath), {encoding:'utf8'})
    if (!data) throw 'given path ['+fpath+'] not found!'
	var lines = data.split('\n')
    if (!lines || !lines.length) throw 'empty dataset?'

    for(var i=0,l=lines.length,arr; i<l; i++){
        arr = lines[i].split(',')
        if (!arr || 2>arr.length) break
        targets.push(IRIS_CLASS[arr.splice(-1, 1)[0]])
        inputs.push(arr.map(toNum))
    }
}
