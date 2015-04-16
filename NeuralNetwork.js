// ref: http://www.cs.bham.ac.uk/~jxb/NN/nn.html
const
E = Math.E,
Pow = Math.pow,
Random = Math.random,
Floor = Math.floor,
Log = Math.log

var
sigmoid = function(i){ return 1/(1+Pow(E, -i)) },
derivative = function(s){ return s*(1-s) },
wrand = function(smallwt){ return 2 * (Random()-0.5) * smallwt },
sse = function(d){return 0.5*d*d}, // sum square error, d = target-output
cee = function(t, o){return t*Log(o)+((1-t)*Log(1-o))}, // cross entropy error
dSSE = function(t, o){return (t-o)*o*(1-o)},
dCEE = function(t, o){return t-o},
NeuralNetwork = function(arg){
    if ('object' !== typeof arg){
        throw 'wrong parameter: '+JSON.stringify(arguments)
    }else if (arg.length){
        this.reboot.apply(this, arguments)
    }else{
        this.recall.apply(this, arguments)
    }
}

NeuralNetwork.prototype = {
    reboot: function(size, config){
        config = config || {}

        var
        layerSize = new Array(size.length),
        layers = [],
        dLayers = [],
        dWeights = [],
        weights = [],
        smallwt = config.smallwt || 0.5,
        i,l,j,jl,k,kl,w,dw,wj,dwj

        // initialize layer size and layers
        for(i=0,l=size.length; i<l; i++){
            jl = size[i]+1
            layerSize[i] = jl
            layers.push(new Array(jl))
            layers.push(new Array(jl))
            dLayers.push(new Array(jl))
        }

        // initialize weights and dweights
        for(i=0,l=layerSize.length-1; i<l; i++){
            jl=layerSize[i+1]
            kl=layerSize[i]
            weights.push(new Array(jl))
            dWeights.push(new Array(jl))
            w = weights[i]
            dw = dWeights[i]
            for(j=1; j<jl; j++){ // start from 1 because the first node's value is always 1
                w[j] = new Array(kl)
                dw[j] = new Array(kl)
                wj = w[j]
                dwj = dw[j]
                for(k=0; k<kl; k++){
                    wj[k]=wrand(smallwt)
                    dwj[k]=0
                }
            }
        }

        this.eta = config.eta || 0.5
        this.alpha = config.alpha || 0.9
        this.layerSize = layerSize.slice()
        this.layers = layers
        this.dLayers = dLayers
        this.weights = weights
        this.dWeights = dWeights
    },
    learn: function(loop, inputs, targets){
        var
        ranpat = [],
        layerSize = this.layerSize,
        layers = this.layers,
        dLayers = this.dLayers,
        weights = this.weights,
        dWeights = this.dWeights,
        eta = this.eta,
        alpha = this.alpha,
        idx,error,target,output,
        i,l,p,j,jl,k,kl,m,ml,wj,wk,dl,dl,dw,dwj,ll,t,o

        for(i=0,l=inputs.length; i<l; i++) ranpat[i]=i

        for(var epoch=0; epoch<loop; epoch++){
            for(l=inputs.length; l; l--, i=Floor(Random()*l), p=ranpat[l], ranpat[l]=ranpat[i], ranpat[i]=p);
            error=0

            for(i=0,l=ranpat.length; i<l; i++){
                idx=ranpat[i]
                target = targets[idx]
                output=this.think(inputs[idx])

                dl=dLayers[layerSize.length-1]
                for(j=0,jl=target.length; j<jl; j++){
                    t=target[j]
                    o=output[j]
                    error += sse(t-o)
                    dl[j+1]=dSSE(t, o)
                }
console.log('dLayers', dLayers)

                // back-propagate
                for(j=layerSize.length-2; j; j--){
                    kl=layerSize[j-1]
                    ml=layerSize[j]
                    dl=dLayers[j]
                    dlj=dLayers[j-1]
                    ll=layers[j]
                    wj=weights[j]
console.log('#################',weights)
console.log(j, kl, ml, dl, dlj, ll, wj)

                    for(k=1; k<kl; k++){ // lower 
                        wk=wj[k]
console.log(wk)
                        t=0
                        for(m=1; m<ml; m++){ // upper 
                            t+=wk[m]*dl[m]
                        }
                        dlj[k]=t*ll[k]*(1-ll[k])
                    }
console.log('dLayers'+k, dLayers)
                }
console.log('dLayers final', dLayers)

                // update weights
                for(j=0,jl=layerSize.length; j<jl; j++){
                    kl=layerSize[j+1]
                    ml=layerSize[j]
                    dw=dWeights[j+1]
                    wj=weights[j+1]
                    dl=dLayers[j+1]
                    ll=layers[j+1]
                    for(k=1; k<kl; k++){ // upper
                        dwj=dw[k]
                        wk=wj[k]
                        dlj=dj[k]
                        dwj[0] = eta*dl[k] + alpha*dwj[0]
                        wk[0]+=dwj[0]
                        for(m=1; m<ml; m++){ // lower
                            dwj[m] = eta*dl[k] + alpha*dwj[m]
                            wk[m]+=dwj[m]
                        }
                    }
                }
            }
            if (error < 0.0004) return error
        }
        return error
    },
    think: function(input, filter){
        var
        layerSize = this.layerSize,
        layers = this.layers,
        weights = this.weights,
        li,lo,loo,wi,wj,
        i,l,j,jl,k,kl

        li = layers[0]
        for(i=0,l=input.length; i<l; i++) li[i+1] = input[i]

        lo = layers[1]
        // apply input filter here
        for(i=0,l=li.length; i<l; i++) lo[i] = filter && filter[i-1] ? filter[i-1]*li[i] : li[i]

        for(i=0,l=layerSize.length-1; i<l; i++){
            jl = layerSize[i+1]
            kl = layerSize[i]
            li=layers[1+(i*2)]
            lo=layers[(i+1)*2]
            loo=layers[1+((i+1)*2)]
            wi=weights[i]
            li[0]=1//first node is always 1
            for(j=1; j<jl; j++){
                wj=wi[j]
                lo[j]=0
                for(k=0; k<kl; k++){
                    lo[j]+=wj[k]*li[k]
                }
                loo[j] = sigmoid(lo[j])
            }
        }
        return layers[layers.length-1].slice(1)
    },
    memory: function(){
        return {
            eta: this.eta,
            alpha: this.alpha,
            layerSize: this.layerSize,
            layers: this.layers,
            dLayers: this.dLayers,
            weights: this.weights,
            dWeights: this.dWeights
        }
    },
    recall: function(memory){
        this.eta = memory.eta
        this.alpha = memory.alpha
        this.layerSize = memory.layerSize
        this.layers = memory.layers
        this.dLayers = memory.dLayers
        this.weights = memory.weights
        this.dWeights = memory.dWeights
    }
}

module.exports = NeuralNetwork
