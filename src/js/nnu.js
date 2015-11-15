/**
 * Get sign, exponent and mantissa from a number.
 * Based on http://stackoverflow.com/questions/9383593/extracting-the-exponent-and-mantissa-of-a-javascript-number
 *
 * @param {number} x Number to convert.
 * @return {Object} with fields sign, exponent and mantissa.
 * Mantissa is returned in the range [1.0, 2.0[ for normal numbers
 * and [0.0, 1.0[ for subnormal numbers or zero.
 */
var getNumberParts = function(x) {
  var float = new Float64Array([x]);
  var bytes = new Uint8Array(float.buffer);
  
  var sign = bytes[7] >> 7;
  var exponent = ((bytes[7] & 0x7f) << 4 | bytes[6] >> 4) - 0x3ff;
  var fraction = 
  // Set the exponent to 0 (exponent bits to match bias)
  bytes[7] = 0x3f;
  bytes[6] |= 0xf0;
  
  return {
    sign: sign,
    exponent: exponent,
    mantissa: float[0]
  };
};

/**
 * Round the parameter as if it was stored to a floating point representation
 * that has the specified bit counts for mantissa and exponent. Works for
 * formats up to 8 exponent bits and 23 mantissa bits.
 *
 * @param {number} src Number to convert.
 * @param {number} mantissaBits How many bits to use for mantissa.
 * @param {number} exponentBits How many bits to use for exponent.
 * @param {boolean} clampToInf Set true to clamp to infinity (instead of the maximum or minimum value supported)
 * @param {boolean} flushSubnormal Set true to flush subnormal numbers to 0 (instead of keeping subnormal values)
 * @return {number} Converted number.
 */
var froundBits = function(src, mantissaBits, exponentBits, clampToInf, flushSubnormal) {
    if (mantissaBits > 23 || exponentBits > 8) {
        return NaN; // Too many bits to simulate!
    }
    if (isNaN(src)) {
        return NaN;
    }

    // Note that Math.pow is specified to return an implementation-dependent approximation,
    // but works well enough in practice to be used here for powers of two.
    var possibleMantissas = Math.pow(2, mantissaBits);
    var mantissaMax = 2.0 - 1.0 / possibleMantissas;
    var max = Math.pow(2, maxNormalExponent(exponentBits)) * mantissaMax; // value with all exponent bits 1 is special
    if (src > max) {
        if (clampToInf) {
            return Infinity;
        } else {
            return max;
        }
    }
    if (src < -max) {
        if (clampToInf) {
            return -Infinity;
        } else {
            return -max;
        }
    }

    var parts = getNumberParts(src);
    // TODO: Customizable rounding (this is now round-to-zero)
    var mantissaRounded = Math.floor(parts.mantissa * possibleMantissas) / possibleMantissas;
    if (parts.exponent + exponentBias(exponentBits) <= 0) {
        if (flushSubnormal) {
            return (parts.sign ? -0 : 0);
        } else {
            while (parts.exponent + exponentBias(exponentBits) <= 0) {
                parts.exponent += 1;
                mantissaRounded = Math.floor(mantissaRounded / 2 * possibleMantissas) / possibleMantissas;
                if (mantissaRounded === 0) {
                    return (parts.sign ? -0 : 0);
                }
            }
        }
    }
    console.log(parts.mantissa);
    console.log(mantissaRounded);
    var x1 = new Float64Array([parts.mantissa]);
    var y1 = new Uint8Array(x1.buffer);
    console.log(y1);
    var x2 = new Float64Array([mantissaRounded]);
    var y2 = new Uint8Array(x2.buffer);
    console.log(y2);
    return (parts.sign ? -1 : 1) * Math.pow(2, parts.exponent) * mantissaRounded;
};

var exponentBias = function(exponentBits) {
    var possibleExponents = Math.pow(2, exponentBits);
    return possibleExponents / 2 - 1;
};

var maxNormalExponent = function(exponentBits) {
    var possibleExponents = Math.pow(2, exponentBits);
    var bias = exponentBias(exponentBits);
    var allExponentBitsOne = possibleExponents - 1;
    return (allExponentBitsOne - 1) - bias;
};

var numberToHalf = function(x){
  return froundBits(x, 23, 8, true, true);
};

// var a = new Uint8Array([0,0,0,0,0,0,255,63])
// var afloat = new Float64Array(a.buffer)
// console.log('afloat: ',afloat);
//
// console.log(numberToHalf(-0.9999999999999999));

var printFloat64ArrayBit = function(farray){
  var uint32a = new Uint16Array(farray.buffer)
  var str = '';
  for(var i=0;i<32;i++){
    str += ((uint32a[1] << i) >>> 31)
  }
  for(var i=0;i<32;i++){
    str += ((uint32a[0] << i) >>> 31)
  }
  return str;
};

var floatUint8 = function(f){
  var fa = new Float64Array([f])
  var ua = new Uint32Array(fa.buffer)
  var u;
  if(f < 0){
    ua[1] = ua[1] & 0x7fffffff
    u = 0x80
  }
  fa[0] += 1
  u |= (ua[1] & 0x000fe000) >> 13  
  return u
}

var uint8Float = function(u){
  var ua = new Uint32Array([0,u])
  ua[1] = ua[1] << 13 | 0x3ff00000;
  var fa = new Float64Array(ua.buffer)
  fa[0] -= 1
  if(u>127){
    ua[1] = ua[1] | 0x80000000
  }else{
    ua[1] = ua[1] & 0x7fffffff
  }
  return fa[0]
}

// for(var i=0;i<256;i++){
//   console.log(floatUint8(uint8Float(i)));
// }
function sortWithIndeces(toSort) {
  for (var i = 0; i < toSort.length; i++) {
    toSort[i] = [toSort[i], i];
  }
  toSort.sort(function(left, right) {
    return left[0] < right[0] ? -1 : 1;
  });
  toSort.sortIndices = [];
  for (var j = 0; j < toSort.length; j++) {
    toSort.sortIndices.push(toSort[j][1]);
    toSort[j] = toSort[j][0];
  }
  return toSort;
}

var buildIndex = function(VD,b){
  var numAtoms = VD.length;
  var numTables = VD[0].length
  var tables = new Array(numTables);
  for(var i=0;i<numTables;i++){
    var table = new Array(256);
    for(var j=0;j<256;j++){
      var vx = uint8Float(j)      
      var diff = new Array(numAtoms);
      for(var k=0;k<numAtoms;k++){
        diff[k] = Math.abs(vx - VD[k][i]);
      }
      var result = sortWithIndeces(diff);
      console.log(result);
      table[j] = result.sortIndices.slice(0,b);
    }
    tables[i] = table;
  }
  return tables;
}
var FastSet = require("collections/fast-set");
var nns = function(VD, Vx){
  var setLen = VD.length;
  var numTables = Vx.length;
  var dist = new Array(setLen);
  
  for(var k=0;k<150;k++){
  for(var i=0;i<setLen;i++){
    var sum = 0;
    for(var j=0;j<numTables;j++){    
      sum += Math.abs(VD[i][j] - Vx[j])
    }
    dist[i] = sum;
  }
  }
  console.log(sortWithIndeces(dist));
  return sortWithIndeces(dist).sortIndices[0];
}
var nnu = function(tables, VD, Vx, numTables){
  // var numTables = Vx.length;
  var candidates = new FastSet();
  for(var i=0;i<numTables;i++){    
    candidates.addEach(tables[i][floatUint8(Vx[i])])
  }
  var candidatesArray = candidates.toArray();
  var setLen = candidatesArray.length;
  console.log(setLen,candidatesArray);
  var dist = new Array(setLen);
  for(var i=0;i<setLen;i++){
    var sum = 0;
    for(var j=0;j<numTables;j++){    
      sum += Math.abs(VD[candidatesArray[i]][j] - Vx[j])
    }
    dist[i] = sum;
  }
  return candidatesArray[sortWithIndeces(dist).sortIndices[0]];
};

var fs = require('fs');
var VD = JSON.parse(fs.readFileSync('VD.json'));
var Vx = VD[200];

var tt = require('tictoc');

// tt.tic();
// var tables = buildIndex(VD, 2);
// tt.toc();
// fs.writeFileSync('tables.json',JSON.stringify(tables));

var tables = JSON.parse(fs.readFileSync('tables.json'));
// console.log(tables);
tt.tic();
console.log(nnu(tables, VD, Vx, 2))
tt.toc();

tt.tic();
console.log(nns(VD, Vx))
tt.toc();

console.log(Vx);
// a = new Float64Array([0.875])
//
// console.log(printFloat64ArrayBit(a))
