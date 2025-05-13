import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import * as TWEEN from '@tweenjs/tween.js'
import { terrainShader } from './shaders/terrain-shader'
import { GUI } from 'dat.gui'
import { Mesh } from 'three'
import axios from 'axios'
import {
    metaState,
    init,
    sessionData,
    initVis,
    gameState,
    logMyState,
    getLocalCordinate,
    readstateFile,
    toggleAnnoation,
} from './util'
import { terrainDimensions } from './constants'
import './styles/style.css'
import * as tiff from 'tiff'
import Stats from 'three/examples/jsm/libs/stats.module'
import { ajax } from 'jquery'
import { url } from 'inspector'
const UPNG = require('upng-js');


// worker.postMessage('i am in worker')
let Developer = false
let overRideControl = false
var data : { [x : number]: Array<number> } = {}

var regionBounds : Array<number> = [0, 0, 0, 0]
var regionDimensions : Array<number> = [0, 0]

let _fetchData: any
let mesh: THREE.Mesh

let isSegmentationDone = true
let isSTLDone = true
let isModelLoaded = false
let isSatelliteImageLoaded = false

const scene = new THREE.Scene()
// const blurs = [0, 1, 2];
// const zs = [100, 200, 300, 400, 500];

const pers = [0, 0.02, 0.04, 0.08, 0.16, 0.32]
// const pers = [0]
var meshes: { [key: string]: Mesh } = {}

var forestArray: Uint8Array

interface PixelDict {
    [position: number]: number;
  }

const pixelDict: PixelDict = {};
var confidenceJSON: PixelDict

var metrices: any
// var forestJson: any
// var gtJson: any

let host = ''
if (location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '172.28.200.135') {
    host = ''
} else {
    host = 'https://floodmap.b-cdn.net/'
}

const stats_mb = Stats()
stats_mb.showPanel(2)
stats_mb.domElement.style.cssText = 'position:absolute;top:250px;right:50px;'
document.body.appendChild(stats_mb.domElement)

const urlParams = new URLSearchParams(window.location.search);
const useParams = urlParams.get('param');
const acq_func = urlParams.get('func');
const use_cod = urlParams.get('cod');
const transform_agg = urlParams.get('t_agg');
const pixel_agg = urlParams.get('p_agg');
const sc_loss = urlParams.get('sc_loss')
const cod_loss = urlParams.get('cod_loss')
const use_forest = urlParams.get('use_forest')

console.log(useParams, acq_func, transform_agg, pixel_agg);

let eventFunction: { [key: string]: any } = {
    BFS: (x: number, y: number, flood: boolean, clear: boolean) => BFSHandler(x, y, flood, clear),
    brush: (x: number, y: number, flood: boolean, clear: boolean) =>
        brushHandler('t', x, y, flood, clear),
    brushLine: (x: number, y: number, flood: boolean, clear: boolean, linePoints: Array<number>) =>
        brushLineHandler(linePoints, flood, clear),
    polygonSelector: (x: number, y: number, flood: boolean, clear: boolean) =>
        polygonSelectionHandler(x, y, flood, clear),
    polygonFill: (
        x: number,
        y: number,
        flood: boolean,
        clear: boolean,
        linePoints: Array<number>
    ) => polygonFillHandler(flood, clear, linePoints),
    segmentation: (x: number, y: number, flood: boolean, clear: boolean) =>
        segAnnotationHandler('s', x, y, flood, clear),
    connectedSegmentation: (x: number, y: number, flood: boolean, clear: boolean) =>
        connectedSegAnnotationHandler('s', x, y, flood, clear),
}

function delay(time: number) {
    return new Promise((resolve) => setTimeout(resolve, time))
}

let time: Date | undefined = undefined

let _readstateFile = async (array: any[]) => {
    sessionData.sessionStart = new Date(array[0].start.time)
    for (let i = 0; i < array.length; i++) {
        if (array[i].start) {
            gameState.push({ start: array[i].start })
            continue
        }
        let event = array[i].mouseEvent
        // if (event.label != "brush") {
        //     await delay(50)
        // }
        // if (i % 1000 == 0) {
        //     console.log(i / array.length)
        // }
        // let _cameraPosition = event.cameraPosition
        // let _target = event.targetPosition
        // camera.position.set(_cameraPosition.x, _cameraPosition.y, _cameraPosition.z)
        // controls.target.set(_target.x, _target.y, _target.z)
        // controls.update()
        let x, y, flood, clear
        if (event.x == undefined) {
            x = 0
            y = 0
        } else {
            x = event.x
            y = event.y
        }
        flood = event.flood
        clear = event.clear
        if (event.brushSize) {
            params.brushSize = event.brushSize
        }
        if (event.persistanceThreshold) {
            params.pers = event.persistanceThreshold
        }
        time = event.time
        eventFunction[event.label](x, y, flood, clear, event.linePoints)
    }
    time = undefined
}
const persLoader = new THREE.TextureLoader()


window.onload = init

var segsToPixels2: {
    [key: number]: {
        [key: number]: Array<number>
    }
} = {}
var persDatas: {
    [key: number]: Int16Array
} = {}

var persTextures: { [key: number]: THREE.Texture } = {}
var dataTextures: { [key: number]: THREE.Texture } = {}
var segsMax: { [key: number]: number } = {}

;(document.getElementById('loader') as HTMLElement).style.display = 'none'
;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
;(document.getElementById('metrices') as HTMLElement).style.display = 'none'
// ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
persLoader.load(
    './img/rainbow.png',
    function (texture) {
        uniforms.colormap.value = texture
    },
    undefined,
    function (err) {
        console.error('An error happened.')
    }
)
// const light = new THREE.SpotLight()
// light.position.set(4000, 4000, 20)
// scene.add(light)
// const ambient = new THREE.AmbientLight( 0x404040 ); // soft white light
// scene.add( ambient );

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 5000)
camera.position.set(0, 0, 2000)

const renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true })
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.shadowMap.enabled = true

document.body.appendChild(renderer.domElement)

let controls = new OrbitControls(camera, renderer.domElement)
controls.dampingFactor = 1.25
controls.enableDamping = true
controls.maxPolarAngle = Math.PI / 1.5
controls.minPolarAngle = 1.2
controls.minDistance = 0
controls.maxAzimuthAngle = 0.8
controls.minAzimuthAngle = -0.65

var canvas = document.createElement('canvas')
var annCanvas = document.createElement('canvas')
var predCanvas = document.createElement('canvas')
var superpixelCanvas = document.createElement('canvas')
var confidenceCanvas = document.createElement('canvas')

var context : CanvasRenderingContext2D
var predContext : CanvasRenderingContext2D
var superpixelContext: CanvasRenderingContext2D
var confidenceContext: CanvasRenderingContext2D
var annotationTexture : THREE.Texture

var predictionTexture: THREE.Texture // 
var superpixelTexture: THREE.Texture // 
var confidenceTexture: THREE.Texture // 

const gui = new GUI({ width: window.innerWidth / 6 })
var params = {
    blur: 0,
    dimension: metaState.flat == 0,
    annotation: true,
    brushSize: 8,
    pers: 6,
    persShow: false,
    data: false,
    guide: 0,
    flood: true,
    dry: false,
    clear: false,
    prediction: false, // 
    superpixel: false, // 
    confidence: false, // 
    predictionKey: 0, // 
    superpixelKey: 0, // 
    entropy: false,
    probability: true,
    cod: false,
    avgTransformation: true,
    minTransformation: false,
    maxTransformation: false,
    avgSuperpixel: true,
    minSuperpixel: false,
    maxSuperpixel: false,
}
let persIndex: { [key: number]: number } = {
    1: 0.32,
    2: 0.16,
    3: 0.08,
    4: 0.04,
    5: 0.02,
    6: 0,
}
var persVal = persIndex[params.pers]
// var persIndex = persToIndex[params.pers];

var uniforms = {
    z: { value: metaState.flat == 0 ? 500 : 0 },
    diffuseTexture: { type: 't', value: new THREE.Texture() },
    annotationTexture: { type: 't', value: new THREE.Texture() },
    predictionTexture: { type: 't', value: new THREE.Texture() },
    superpixelTexture: { type: 't', value: new THREE.Texture() },
    confidenceTexture: { type: 't', value: new THREE.Texture() },
    dataTexture: { type: 't', value: new THREE.Texture() },
    persTexture: { type: 't', value: new THREE.Texture() },
    colormap: { type: 't', value: new THREE.Texture() },
    annotation: { value: 1 },
    prediction: {value: 0},
    superpixel: {value: 0},
    confidence: {value: 0},
    predictionKey: {value: 0},
    superpixelKey: {value: 0},
    data: { value: 0 },
    segsMax: { type: 'f', value: 0 },
    persShow: { value: 0 },
    hoverValue: { type: 'f', value: 0 },
    guide: { value: params.guide },
    dimensions: { type: 'vec2', value: [100, 100] },
    dry: { type: 'bool', value: params.dry },
    flood: { type: 'bool', value: params.flood },
    quadrant: { value: metaState.quadrant },
    entropy: { value: 0 },
    probability: { value: 0},
    cod: { value: 0 },
    avgTransformation: {value: 1},
    minTransformation: {value: 0},
    maxTransformation: {value: 0},
    avgSuperpixel: {value: 1},
    minSuperpixel: {value: 0},
    maxSuperpixel: {value: 0},
    sc_loss: {value: 1},
    cod_loss: {value: 1},
    use_forest: {value: 1}
}
const viewFolder = gui.addFolder('Settings')
const scFolder = gui.addFolder('Uncertainty Measure')
const transformationFolder = gui.addFolder('Transform-level Agg')
const superpixelFolder = gui.addFolder('Pixel-level Agg')


viewFolder
    .add(params, 'annotation')
    .onChange(() => {
        if (params.annotation) {
            uniforms.annotation.value = 1
        } else {
            uniforms.annotation.value = 0
        }
    })
    .name('Show Annotation')
// 
viewFolder
.add(params, 'prediction')
.onChange(() => {
    if (params.prediction) {
        uniforms.prediction.value = 1
    } else {
        uniforms.prediction.value = 0
    }
})
.name('Show Prediction')
//

// 
viewFolder
.add(params, 'superpixel')
.onChange(() => {
    if (params.superpixel) {
        uniforms.superpixel.value = 1
    } else {
        uniforms.superpixel.value = 0
    }
})
.name('Show Superpixels')
//

// 
viewFolder
.add(params, 'confidence')
.onChange(() => {
    if (params.confidence) {
        uniforms.confidence.value = 1
    } else {
        uniforms.confidence.value = 0
    }
})
.name('Show Forest Prediction')
//


let sizeMap = {
    brushSize: {
        '4x4': 4,
        '8x8': 8,
        '16x16': 16,
        '32x32': 32,
    },
}

viewFolder
    .add(sizeMap, 'brushSize', sizeMap.brushSize)
    .setValue(8)
    .onChange((value) => {
        params.brushSize = value
    })
    .name('Brush Size')

viewFolder
    .add(
        {
            x: () => {
                new TWEEN.Tween(controls.target)
                    .to(
                        {
                            x: (regionBounds[1] + regionBounds[0]) / 2,
                            y: (regionBounds[2] + regionBounds[3]) / 2,
                            z: 0,
                        },
                        1000
                    )
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onUpdate(() => {
                        controls.update()
                    })
                    .start()
                new TWEEN.Tween(camera.position)
                    .to(
                        {
                            x: (regionBounds[1] + regionBounds[0]) / 2,
                            y: (regionBounds[2] + regionBounds[3]) / 2,
                            z: 2000,
                        },
                        1000
                    )
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onUpdate(() => {
                        camera.updateProjectionMatrix()
                    })
                    .start()
            },
        },
        'x'
    )
    .name('Reset Camera View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(-500, regionDimensions[1] / 2, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Left View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] + 500, regionDimensions[1] / 2, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Right View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] / 2, regionDimensions[1] + 500, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Top View')
// viewFolder
//     .add(
//         {
//             x: () => {
//                 camera.position.set(regionDimensions[0] / 2, -500, 500)
//                 camera.up.set(0, 0, 1)
//                 controls.dispose()
//                 controls = new OrbitControls(camera, renderer.domElement)
//                 controls.target = new THREE.Vector3(
//                     regionDimensions[0] / 2,
//                     regionDimensions[1] / 2,
//                     -1000
//                 )
//             },
//         },
//         'x'
//     )
//     .name('Camera to Bottom View')

viewFolder.open()
// meshFolder.open()

// scFolder
//     .add(params, 'entropy')
//     .onChange(() => {
//         if (params.entropy) {
//             uniforms.entropy.value = 1
//         } else {
//             uniforms.entropy.value = 0
//         }
//     })
//     .name('Entropy')

// scFolder
//     .add(params, 'probability')
//     .onChange(() => {
//         if (params.probability) {
//             uniforms.probability.value = 1
//         } else {
//             uniforms.probability.value = 0
//         }
//     })
//     .name('Probability')

const checkboxValues = {
    probability: true,
    entropy: false,
    cod: false
};

const checkboxValuesTransformation = {
    avgTransformation: true,
    minTransformation: false,
    maxTransformation: false
};

const checkboxValuesSuperpixel = {
    avgSuperpixel: true,
    minSuperpixel: false,
    maxSuperpixel: false
};


// Add checkboxes to the folder
const probability = scFolder.add(checkboxValues, 'probability').name('Probability');
const entropy = scFolder.add(checkboxValues, 'entropy').name('Entropy');
const cod = scFolder.add(checkboxValues, 'cod').name('COD');

// Add checkboxes to the folder
const avgTransformation = transformationFolder.add(checkboxValuesTransformation, 'avgTransformation').name('Average');
// const minTransformation = transformationFolder.add(checkboxValuesTransformation, 'minTransformation').name('Minimum');
const maxTransformation = transformationFolder.add(checkboxValuesTransformation, 'maxTransformation').name('Maximum');

// Add checkboxes to the folder
const avgSuperpixel = superpixelFolder.add(checkboxValuesSuperpixel, 'avgSuperpixel').name('Average');
const minSuperpixel = superpixelFolder.add(checkboxValuesSuperpixel, 'minSuperpixel').name('Minimum');
// const maxSuperpixel = superpixelFolder.add(checkboxValuesSuperpixel, 'maxSuperpixel').name('Maximum');

// uniforms.probability.value = +checkboxValues['probability'];
// uniforms.entropy.value = +checkboxValues['entropy'];
// uniforms.cod.value = +checkboxValues['cod'];

if (useParams == '1'){
    if (acq_func == 'prob'){
        probability.setValue(true);
        params.probability = true;
        uniforms.probability.value = 1;

        entropy.setValue(false);
        params.entropy = false;
        uniforms.entropy.value = 0;

        if (use_cod == '1'){
            cod.setValue(true);
            params.cod = true;
            uniforms.cod.value = 1;
        }
        else{
            cod.setValue(false);
            params.cod = false;
            uniforms.cod.value = 0;
        }
    }
    else if (acq_func == 'ent'){
        probability.setValue(false);
        params.probability = false;
        uniforms.probability.value = 0;

        entropy.setValue(true);
        params.entropy = true;
        uniforms.entropy.value = 1;

        if (use_cod == '1'){
            cod.setValue(true);
            params.cod = true;
            uniforms.cod.value = 1;
        }
        else{
            cod.setValue(false);
            params.cod = false;
            uniforms.cod.value = 0;
        }
    }

    if (transform_agg == 'avg'){
        avgTransformation.setValue(true);
        params.avgTransformation = true;
        uniforms.avgTransformation.value = 1;

        maxTransformation.setValue(false);
        params.maxTransformation = false;
        uniforms.maxTransformation.value = 0;
    }
    else if (transform_agg == 'max'){
        avgTransformation.setValue(false);
        params.avgTransformation = false;
        uniforms.avgTransformation.value = 0;

        maxTransformation.setValue(true);
        params.maxTransformation = true;
        uniforms.maxTransformation.value = 1;
    }

    if (pixel_agg == 'avg'){
        avgSuperpixel.setValue(true);
        params.avgSuperpixel = true;
        uniforms.avgSuperpixel.value = 1;

        minSuperpixel.setValue(false);
        params.minSuperpixel = false;
        uniforms.minSuperpixel.value = 0;
    }
    else if (pixel_agg == 'min'){
        avgSuperpixel.setValue(false);
        params.avgSuperpixel = false;
        uniforms.avgSuperpixel.value = 0;

        minSuperpixel.setValue(true);
        params.minSuperpixel = true;
        uniforms.minSuperpixel.value = 1;
    }
}

if (sc_loss == '0'){
    uniforms.sc_loss.value = 0;
}
else if (sc_loss == '1'){
    uniforms.sc_loss.value = 1;
}

if (cod_loss == '0'){
    uniforms.cod_loss.value = 0;
}
else if (cod_loss == '1'){
    uniforms.cod_loss.value = 1;
}

if (use_forest == '0'){
    uniforms.use_forest.value = 0;
}
else if (use_forest == '1'){
    uniforms.use_forest.value = 1;
}
    


console.log("uniforms.probab: ", uniforms.probability.value)
console.log("uniforms.ent: ", uniforms.entropy.value)
console.log("uniforms.cod: ", uniforms.cod.value)

console.log("uniforms.avgTransformation: ", uniforms.avgTransformation.value)
console.log("uniforms.maxTransformation: ", uniforms.maxTransformation.value)
console.log("uniforms.avgSuperpixel: ", uniforms.avgSuperpixel.value)
console.log("uniforms.minSuperpixel: ", uniforms.minSuperpixel.value)

console.log("sc_loss: ", uniforms.sc_loss.value)
console.log("cod_loss: ", uniforms.cod_loss.value)


// Set up the mutual exclusivity behavior
probability.onChange(function (value) {
    if (useParams == '1'){
        if (acq_func == 'prob'){
            console.log("acq_func: ", 'prob');
            value = 1;
        }
        else{
            value = 0;
        }
    } 
      
    if (value) {
        entropy.setValue(false);
        params.probability = true;
        uniforms.probability.value = 1;
        params.entropy = false;
        uniforms.entropy.value = 0;

        // maxSuperpixel.domElement.style.opacity = "0.1"
        // maxTransformation.domElement.style.opacity = "0.1";
    }
    else{
        // maxSuperpixel.domElement.style.opacity = "1";
        maxTransformation.domElement.style.opacity = "1";
    }
    
});

entropy.onChange(function (value) {
    if (useParams == '1'){
        if (acq_func == 'ent'){
            console.log("acq_func: ", 'ent');
            value = 1;
        }
        else{
            value = 0;
        }
    } 

    if (value) {
        probability.setValue(false);
        params.entropy = true;
        uniforms.entropy.value = 1;
        params.probability = false;
        uniforms.probability.value = 0;

        // minSuperpixel.domElement.style.opacity = "0.1"
        // minTransformation.domElement.style.opacity = "0.1";
    } else {
        minSuperpixel.domElement.style.opacity = "1";
        // minTransformation.domElement.style.opacity = "1";
    }
});

cod.onChange(function (value) {
    if (useParams == '1'){
        if (use_cod == '1'){
            console.log("cod: ", 'cod');
            value = 1;
        }
        else{
            value = 0;
        }
    } 
    
    if (value) {
        params.cod = true;
        uniforms.cod.value = 1;
    }
});

scFolder.open()

// Set up the mutual exclusivity behavior
avgTransformation.onChange(function (value) {
    if (useParams == '1'){
        if (transform_agg == 'avg'){
            console.log("transform_agg: ", 'avg');
            value = 1;
        }
        else{
            value = 0;
        }
    }

    if (value) {
        // minTransformation.setValue(false);
        params.avgTransformation = true;
        uniforms.avgTransformation.value = 1;
        // params.minTransformation = false;
        // uniforms.minTransformation.value = 0;

        maxTransformation.setValue(false);
        params.maxTransformation = false;
        uniforms.maxTransformation.value = 0;
    }
    
});

// minTransformation.onChange(function (value) {
//     if (value) {
//         if (params.probability == true){
//             avgTransformation.setValue(false);
//             params.minTransformation = true;
//             uniforms.minTransformation.value = 1;
//             params.avgTransformation = false;
//             uniforms.avgTransformation.value = 0;

//             maxTransformation.setValue(false);
//             params.maxTransformation = false;
//             uniforms.maxTransformation.value = 0;
//         }
//         else if (params.entropy == true){
//             minTransformation.setValue(false);
//             params.minTransformation = false;
//             uniforms.minTransformation.value = 0;
//         }
        
//     }
// });

maxTransformation.onChange(function (value) {
    if (useParams == '1'){
        if (transform_agg == 'max'){
            console.log("transform_agg: ", 'max');
            value = 1;
        }
        else{
            value = 0;
        }
    }

    if (value) {
        // if (params.entropy == true){
        //     avgTransformation.setValue(false);
        //     // minTransformation.setValue(false);
        //     params.avgTransformation = false;
        //     uniforms.avgTransformation.value = 0;

        //     params.maxTransformation = true;
        //     uniforms.maxTransformation.value = 1;

        //     // params.minTransformation = false;
        //     // uniforms.minTransformation.value = 0;
        // }
        // else if (params.probability == true){
        //     maxTransformation.setValue(false);
        //     params.maxTransformation = false;
        //     uniforms.maxTransformation.value = 0;
        // }

        avgTransformation.setValue(false);
        // minTransformation.setValue(false);
        params.avgTransformation = false;
        uniforms.avgTransformation.value = 0;

        params.maxTransformation = true;
        uniforms.maxTransformation.value = 1;

        // params.minTransformation = false;
        // uniforms.minTransformation.value = 0;

    }
});

transformationFolder.open()


// Set up the mutual exclusivity behavior
avgSuperpixel.onChange(function (value) {
    if (useParams == '1'){
        if (pixel_agg == 'avg'){
            console.log("pixel_agg: ", 'avg');
            value = 1;
        }
        else{
            value = 0;
        }
    }

    if (value) {
        minSuperpixel.setValue(false);
        params.avgSuperpixel = true;
        uniforms.avgSuperpixel.value = 1;
        params.minSuperpixel = false;
        uniforms.minSuperpixel.value = 0;

        // maxSuperpixel.setValue(false);
        // params.maxSuperpixel = false;
        // uniforms.maxSuperpixel.value = 0;
    }
    
});

minSuperpixel.onChange(function (value) {
    if (useParams == '1'){
        if (pixel_agg == 'min'){
            console.log("pixel_agg: ", 'min');
            value = 1;
        }
        else{
            value = 0;
        }
    }

    if (value) {
        // if (params.probability == true){
        //     avgSuperpixel.setValue(false);
        //     params.minSuperpixel = true;
        //     uniforms.minSuperpixel.value = 1;
        //     params.avgSuperpixel = false;
        //     uniforms.avgSuperpixel.value = 0;

        //     // maxSuperpixel.setValue(false);
        //     // params.maxSuperpixel = false;
        //     // uniforms.maxSuperpixel.value = 0;
        // }
        // else if (params.entropy == true){
        //     minSuperpixel.setValue(false);
        //     params.minSuperpixel = false;
        //     uniforms.minSuperpixel.value = 0;
        // }

        avgSuperpixel.setValue(false);
        params.minSuperpixel = true;
        uniforms.minSuperpixel.value = 1;
        params.avgSuperpixel = false;
        uniforms.avgSuperpixel.value = 0;

        // maxSuperpixel.setValue(false);
        // params.maxSuperpixel = false;
        // uniforms.maxSuperpixel.value = 0;
        
    }
});

// maxSuperpixel.onChange(function (value) {
//     if (value) {
//         if (params.entropy == true){
//             avgSuperpixel.setValue(false);
//             minSuperpixel.setValue(false);

//             params.maxSuperpixel = true;
//             uniforms.maxSuperpixel.value = 1;
            
//             params.minSuperpixel = false;
//             uniforms.minSuperpixel.value = 0;

//             params.avgSuperpixel = false;
//             uniforms.avgSuperpixel.value = 0;
//         }
//         else if (params.probability == true){
//             maxSuperpixel.setValue(false);
//             params.maxSuperpixel = false;
//             uniforms.maxSuperpixel.value = 0;
//         }
//     }
// });

superpixelFolder.open()



function segSelect(x: number, y: number, color: string) {
    context!.fillStyle = color
    var value = persDatas[persVal][x + y * regionDimensions[0]]
    var pixels = segsToPixels2[persVal][value]
    for (var i = 0; i < pixels.length; i++) {
        var x = pixels[i] % regionDimensions[0]
        var y = regionDimensions[1] - 1 - Math.floor(pixels[i] / regionDimensions[0])
        if (color == 'clear') {
            context!.clearRect(x, y, 1, 1)
            sessionData.annotatedPixelCount--
        } else {
            context!.fillRect(x, y, 1, 1)
            sessionData.annotatedPixelCount++
        }
    }
    annotationTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // 
}

function connectedSegSelect(x: number, y: number, flood: boolean, clear: boolean) {
    var color = 'blue'
    if (flood) {
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    visited = new Map()
    BFS(x, y, 'BFS_Segment', color, flood)
}

const searchFunction = {
    BFS_Down: {
        E: (x: number, y: number, value: number) => data[persVal][x + 1 + y * regionDimensions[0]] <= value,
        W: (x: number, y: number, value: number) => data[persVal][x - 1 + y * regionDimensions[0]] <= value,
        N: (x: number, y: number, value: number) =>
            data[persVal][x + (y + 1) * regionDimensions[0]] <= value,
        S: (x: number, y: number, value: number) =>
            data[persVal][x + (y - 1) * regionDimensions[0]] <= value,
        EN: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y + 1) * regionDimensions[0]] <= value,
        WN: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y + 1) * regionDimensions[0]] <= value,
        SW: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y - 1) * regionDimensions[0]] <= value,
        SE: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y - 1) * regionDimensions[0]] <= value,
    },
    BFS_Hill: {
        E: (x: number, y: number, value: number) => data[persVal][x + 1 + y * regionDimensions[0]] >= value,
        W: (x: number, y: number, value: number) => data[persVal][x - 1 + y * regionDimensions[0]] >= value,
        N: (x: number, y: number, value: number) =>
            data[persVal][x + (y + 1) * regionDimensions[0]] >= value,
        S: (x: number, y: number, value: number) =>
            data[persVal][x + (y - 1) * regionDimensions[0]] >= value,
        EN: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y + 1) * regionDimensions[0]] >= value,
        WN: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y + 1) * regionDimensions[0]] >= value,
        SW: (x: number, y: number, value: number) =>
            data[persVal][x - 1 + (y - 1) * regionDimensions[0]] >= value,
        SE: (x: number, y: number, value: number) =>
            data[persVal][x + 1 + (y - 1) * regionDimensions[0]] >= value,
    },
    BFS_Segment: {
        E: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + y * regionDimensions[0]] == value,
        W: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + y * regionDimensions[0]] == value,
        N: (x: number, y: number, value: number) =>
            persDatas[persVal][x + (y + 1) * regionDimensions[0]] == value,
        S: (x: number, y: number, value: number) =>
            persDatas[persVal][x + (y - 1) * regionDimensions[0]] == value,
        EN: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + (y + 1) * regionDimensions[0]] == value,
        WN: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + (y + 1) * regionDimensions[0]] == value,
        SW: (x: number, y: number, value: number) =>
            persDatas[persVal][x - 1 + (y - 1) * regionDimensions[0]] == value,
        SE: (x: number, y: number, value: number) =>
            persDatas[persVal][x + 1 + (y - 1) * regionDimensions[0]] == value,
    },
}

const valueFunction = {
    BFS_Down: (x: number, y: number) => data[persVal][x + y * regionDimensions[0]],
    BFS_Hill: (x: number, y: number) => data[persVal][x + y * regionDimensions[0]],
    BFS_Segment: (x: number, y: number) =>
        persDatas[persVal][x + y * regionDimensions[0]],
}

const fillFunction = {
    BFS_Down: (x: number, y: number) => [x, y],
    BFS_Hill: (x: number, y: number) => [x, y],
    BFS_Segment: (x: number, y: number) => [x, y],
}

var visited = new Map()
function BFS(x: number, y: number, direction: string, color: string, flood: boolean) {
    context!.fillStyle = color
    var stack = []
    visited.set(`${x}, ${y}`, 1)
    stack.push(x, y)
    type ObjectKey = keyof typeof searchFunction
    let _direction = direction as ObjectKey
    while (stack.length > 0) {
        y = stack.pop()!
        x = stack.pop()!

        // const pixelIndex = y * regionDimensions[0] + x
        // const pixelVal = gtJson[pixelIndex]

        // var flood_pixel = false;
        // var dry_pixel = false;
        // var unk_pixel = false;

        // if (pixelVal == 1){
        //     flood_pixel = true;
        // }
        // else if (pixelVal == -1){
        //     dry_pixel = true;
        // }
        // else{
        //     unk_pixel = true;
        // }
        
        // var stop_label_propagation;
        // if (pixelVal == 0){
        //     stop_label_propagation = true;
        // }
        // else if (pixelVal != 0){
        //     if (flood == true && flood_pixel == true){
        //         stop_label_propagation = false;
        //     }
        //     else if (flood == false && dry_pixel == true){
        //         stop_label_propagation = false;
        //     }
        //     else{
        //         stop_label_propagation = true;
        //     }
        // }

        if (
            x < regionBounds[0] ||
            x > regionBounds[1] ||
            y < regionBounds[2] ||
            y > regionBounds[3]
        ) {
            continue
        }
        // else if (stop_label_propagation == true){
        //     continue
        // }


        let [fillX, fillY] = fillFunction[_direction](x, y)
        if (color == 'clear') {
            sessionData.annotatedPixelCount--
            context!.clearRect(fillX, fillY, 1, 1)
        } else {
            sessionData.annotatedPixelCount++
            context!.fillRect(fillX, fillY, 1, 1)
        }
        var value = valueFunction[_direction](x, y)
        if (searchFunction[_direction].E(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y}`)) {
                visited.set(`${x + 1}, ${y}`, 1)
                stack.push(x + 1, y)
            }
        }
        if (searchFunction[_direction].W(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y}`)) {
                visited.set(`${x - 1}, ${y}`, 1)
                stack.push(x - 1, y)
            }
        }
        if (searchFunction[_direction].N(x, y, value)) {
            if (!visited.get(`${x}, ${y + 1}`)) {
                visited.set(`${x}, ${y + 1}`, 1)
                stack.push(x, y + 1)
            }
        }
        if (searchFunction[_direction].S(x, y, value)) {
            if (!visited.get(`${x}, ${y - 1}`)) {
                visited.set(`${x}, ${y - 1}`, 1)
                stack.push(x, y - 1)
            }
        }
        if (searchFunction[_direction].EN(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y + 1}`)) {
                visited.set(`${x + 1}, ${y + 1}`, 1)
                stack.push(x + 1, y + 1)
            }
        }
        if (searchFunction[_direction].WN(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y + 1}`)) {
                visited.set(`${x - 1}, ${y + 1}`, 1)
                stack.push(x - 1, y + 1)
            }
        }
        if (searchFunction[_direction].SW(x, y, value)) {
            if (!visited.get(`${x - 1}, ${y - 1}`)) {
                visited.set(`${x - 1}, ${y - 1}`, 1)
                stack.push(x - 1, y - 1)
            }
        }
        if (searchFunction[_direction].SE(x, y, value)) {
            if (!visited.get(`${x + 1}, ${y - 1}`)) {
                visited.set(`${x + 1}, ${y - 1}`, 1)
                stack.push(x + 1, y - 1)
            }
        }
    }
    annotationTexture.needsUpdate = true
    // uniforms.annotationTexture.value = annotationTexture;
}

function fpart(x: number) {
    return x - Math.floor(x)
}
function rfpart(x: number) {
    return 1 - fpart(x)
}

const pointer = new THREE.Vector2()
const raycaster = new THREE.Raycaster()
var skip = true
var skipCounter = 0
const onMouseMove = (event: MouseEvent) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1
    if (skipCounter == 4) {
        skip = false
        skipCounter = 0
    } else {
        skipCounter++
    }
}
var polyPoints: Array<number> = []

function performRayCasting() {
    raycaster.setFromCamera(pointer, camera)
    const intersects = raycaster.intersectObjects(scene.children)
    var point = intersects[0].point
    var x = Math.trunc(point.x)
    var y = Math.ceil(point.y)
    return [x, y]
}

function hoverHandler() {
    let [x, y] = performRayCasting()
    y = regionDimensions[1] - 1 - y
    let localId = persDatas[persVal][x + y * regionDimensions[0]]
    uniforms.hoverValue.value = localId
    params.guide = 1
    uniforms.guide.value = params.guide
}

function buttonPressHandlerSuperpixel() {
    params.superpixelKey = 1
    uniforms.superpixelKey.value = params.superpixelKey
}

function buttonPressHandlerPrediction() {
    params.predictionKey = 1
    uniforms.predictionKey.value = params.predictionKey
}

function BFSHandler(x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    visited = new Map()
    var type = 'BFS_Hill'
    var color = 'blue'
    if (flood) {
        type = 'BFS_Down'
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    BFS(x, y, type, color, flood)
    // logMyState('f', 'BFS', flood, clear, camera, pointer, x, y, undefined, undefined, time)
}

function brushHandler(key: string, x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    context!.fillStyle = 'blue'
    if (flood) {
        context!.fillStyle = 'red'
    }
    if (clear) {
        context!.clearRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )
        sessionData.annotatedPixelCount -= params.brushSize * params.brushSize
    } else {
        context!.fillRect(
            x - Math.floor(params.brushSize / 2),
            y - Math.floor(params.brushSize / 2),
            params.brushSize,
            params.brushSize
        )
        sessionData.annotatedPixelCount += params.brushSize * params.brushSize
    }
    annotationTexture.needsUpdate = true

    // uniforms.annotationTexture.value = annotationTexture
    // logMyState(key, 'brush', flood, clear, camera, pointer, x, y, params.brushSize, undefined, time)
}

function brushLineHandler(linePixels: Array<number>, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    context!.fillStyle = 'blue'
    if (flood) {
        context!.fillStyle = 'red'
    }
    for (var i = 0; i < linePixels.length; i += 2) {
        if (clear) {
            context!.clearRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )
            sessionData.annotatedPixelCount -= params.brushSize * params.brushSize
        } else {
            context!.fillRect(
                linePixels[i] - Math.floor(params.brushSize / 2),
                regionDimensions[1] - 1 - linePixels[i + 1] - Math.floor(params.brushSize / 2),
                params.brushSize,
                params.brushSize
            )
            sessionData.annotatedPixelCount += params.brushSize * params.brushSize
        }
    }
    annotationTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // 
    // logMyState(
    //     't',
    //     'brushLine',
    //     flood,
    //     clear,
    //     camera,
    //     undefined,
    //     undefined,
    //     undefined,
    //     params.brushSize,
    //     linePixels,
    //     time
    // )
}

function polygonSelectionHandler(x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    context!.fillStyle = 'blue'
    if (flood) {
        context!.fillStyle = 'red'
    }
    if (clear) {
        var cy = polyPoints.pop()!
        var cx = polyPoints.pop()!
        context!.clearRect(cx - 2, cy - 2, 4, 4)
        sessionData.annotatedPixelCount -= 16 //follow this with the line selection to minimize the double counting
    } else {
        polyPoints.push(x, y)
        context!.fillRect(x - 2, y - 2, 4, 4)
        sessionData.annotatedPixelCount += 16 //follow this with the line selection to minimize the double counting
    }
    // logMyState(
    //     'p',
    //     'polygonSelector',
    //     flood,
    //     clear,
    //     camera,
    //     pointer,
    //     x,
    //     y,
    //     params.brushSize,
    //     undefined,
    //     time
    // )
    annotationTexture.needsUpdate = true
}

function polygonFillHandler(flood: boolean, clear: boolean, linePoints?: Array<number>) {
    sessionData.numberofClick++
    if (linePoints) {
        polyPoints = linePoints
    }
    var type = 'BFS_Hill'
    var color = 'blue'
    if (flood) {
        color = 'red'
        type = 'BFS_Down'
    }
    context!.fillStyle = color
    context!.beginPath()
    context!.moveTo(polyPoints[0], polyPoints[1])
    for (var i = 2; i < polyPoints.length; i += 2) {
        context!.lineTo(polyPoints[i], polyPoints[i + 1])
        context!.clearRect(polyPoints[i] - 2, polyPoints[i + 1] - 2, 4, 4)
    }
    context!.closePath()
    if (clear) {
        color = 'clear'
        context!.globalCompositeOperation = 'destination-out'
        context!.fill()
        // second pass, the actual painting, with the desired color
        context!.globalCompositeOperation = 'source-over'
        context!.fillStyle = 'rgba(0,0,0,0)'
    }
    context!.fill()
    var linePixels: Array<number> = []
    for (var i = 0; i < polyPoints.length; i += 2) {
        var x0 = polyPoints[i]
        var y0 = polyPoints[i + 1]
        var x1, y1
        if (i + 2 == polyPoints.length) {
            x1 = polyPoints[0]
            y1 = polyPoints[1]
        } else {
            x1 = polyPoints[i + 2]
            y1 = polyPoints[i + 3]
        }
        var steep: boolean = Math.abs(y1 - y0) > Math.abs(x1 - x0)
        if (steep) {
            ;[x0, y0] = [y0, x0]
            ;[x1, y1] = [y1, x1]
        }
        if (x0 > x1) {
            ;[x0, x1] = [x1, x0]
            ;[y0, y1] = [y1, y0]
        }
        var dx = x1 - x0
        var dy = y1 - y0
        var gradient
        if (dx == 0) {
            gradient = 1
        } else {
            gradient = dy / dx
        }
        var xend = x0
        var yend = y0
        var xpxl1 = xend
        var ypxl1 = yend
        if (steep) {
            linePixels.push(ypxl1, xpxl1)
            linePixels.push(ypxl1 + 1, xpxl1)
        } else {
            linePixels.push(xpxl1, ypxl1)
            linePixels.push(xpxl1, ypxl1 + 1)
        }
        var intery = yend + gradient
        xend = x1
        yend = y1
        var xpxl2 = xend
        var ypxl2 = yend
        if (steep) {
            linePixels.push(ypxl2, xpxl2)
            linePixels.push(ypxl2 + 1, xpxl2)
        } else {
            linePixels.push(xpxl2, ypxl2)
            linePixels.push(xpxl2, ypxl2 + 1)
        }
        if (steep) {
            for (var x = xpxl1 + 1; x < xpxl2; x++) {
                linePixels.push(Math.floor(intery), x)
                linePixels.push(Math.floor(intery) + 1, x)
                intery = intery + gradient
            }
        } else {
            for (var x = xpxl1 + 1; x < xpxl2; x++) {
                linePixels.push(x, Math.floor(intery))
                linePixels.push(x, Math.floor(intery) + 1)
                intery = intery + gradient
            }
        }
    }
    visited = new Map()
    for (var i = 0; i < linePixels.length; i += 2) {
        BFS(linePixels[i], linePixels[i + 1], type, color, flood)
    }
    // logMyState(
    //     'o',
    //     'polygonFill',
    //     flood,
    //     clear,
    //     camera,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     polyPoints,
    //     time
    // )
    polyPoints = []
    annotationTexture.needsUpdate = true
    // predictionTexture.needsUpdate = true // 
}

function segAnnotationHandler(key: string, x: number, y: number, flood: boolean, clear: boolean) {
    sessionData.numberofClick++
    var color = 'blue'
    if (flood) {
        color = 'red'
    }
    if (clear) {
        color = 'clear'
    }
    context!.fillStyle = color
    segSelect(x, y, color)
    // logMyState(key, 'segmentation', flood, clear, camera, pointer, x, y, undefined, undefined, time)
}

function connectedSegAnnotationHandler(
    key: string,
    x: number,
    y: number,
    flood: boolean,
    clear: boolean
) {
    sessionData.numberofClick++
    connectedSegSelect(x, y, flood, clear)
    // logMyState(
    //     key,
    //     'connectedSegmentation',
    //     flood,
    //     clear,
    //     camera,
    //     pointer,
    //     x,
    //     y,
    //     undefined,
    //     undefined,
    //     time
    // )
}

// Function to display the cross mark
const crossContainer = document.getElementById('crossContainer');
function displayCrossMark(x: any, y: any) {

    if (crossContainer){
        // Create a div for the cross mark
        // const crossMark = document.createElement('div');
        // crossMark.className = 'crossMark';

        // // Position the cross mark at the clicked coordinates
        // crossMark.style.left = `${x}px`;
        // crossMark.style.top = `${y}px`;

        // console.log("inside crossmark")

        const preElement = document.createElement('pre');

        preElement.style.left = `${x}px`;
        preElement.style.top = `${y}px`;

        // Convert JSON object to a string with indentation
        const jsonString = "X";

        // Set the content of the <pre> element to the formatted JSON string
        preElement.textContent = jsonString;

        // Append the cross mark to the container
        crossContainer.innerHTML = ''; // Clear previous marks
        crossContainer.appendChild(preElement);

        ;(document.getElementById('crossContainer') as HTMLElement).style.display = 'block'
    } else {
        console.log('crossContainer not found.');
    }
}

let [lastX, lastY] = [0, 0]
const onKeyPress = (event: KeyboardEvent) => {
    if (event.key == 'z') {
        console.log("Z pressed")
        var eve
        console.log("ZZ pressed")
        for (var i = gameState.length - 1; i > 0; i--) {
            console.log("Z here")
            if (!gameState[i]['mouseEvent'].undone && !gameState[i]['mouseEvent'].clear) {
                sessionData.numberofUndo++
                gameState[i]['mouseEvent'].undone = true
                eve = gameState[i]['mouseEvent']
                break
            }
        }
        if (eve) {
            eventFunction[eve.label](eve.x, eve.y, eve.flood, !eve.clear, eve.linePoints)
        }
    } else if (event.key == 'r') {
        var eve
        for (var i = gameState.length - 1; i > 0; i--) {
            if (!gameState[i]['mouseEvent'].redone && gameState[i]['mouseEvent'].clear) {
                sessionData.numberofRedo++
                gameState[i]['mouseEvent'].redone = true
                eve = gameState[i]['mouseEvent']
                break
            }
        }
        if (eve) {
            eventFunction[eve.label](eve.x, eve.y, eve.flood, !eve.clear, eve.linePoints)
        }
    }

    if (event.repeat && skip) {
        return
    }
    skip = true

    if (event.key == 'm') {
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('exploration') as HTMLButtonElement).innerHTML = 'Continue ->'
    } else if (event.key == 'g' && metaState.segEnabled) {
        hoverHandler()
    } else if (event.key == 'f' && metaState.BFS) {
        let [x, y_orig] = performRayCasting()

        let y: any = regionDimensions[1] - 1 - y_orig // y is height, x is width
        BFSHandler(x, y, params.flood, params.clear)

        // const pixelIndex = y * regionDimensions[0] + x
        // const pixelVal = gtJson[pixelIndex]

        // var flood_pixel = false;
        // var dry_pixel = false;
        // var unk_pixel = false;

        // if (pixelVal == 1){
        //     flood_pixel = true;
        // }
        // else if (pixelVal == -1){
        //     dry_pixel = true;
        // }
        // else{
        //     unk_pixel = true;
        // }
        
        // if (pixelVal != 0){
        //     if (params.flood == true && flood_pixel == true){
        //         BFSHandler(x, y, params.flood, params.clear)
        //     }
        //     else if (params.flood == false && dry_pixel == true){
        //         BFSHandler(x, y, params.flood, params.clear)
        //     }
        // }
        

        // const pixelIndex = y * regionDimensions[0] + x
        // const pixelVal = forestJson[pixelIndex]
        

        // if (pixelVal == 0){
        //     // y = regionDimensions[1] - 1 - y
        //     BFSHandler(x, y, params.flood, params.clear)
        //     ;(document.getElementById('crossContainer') as HTMLElement).style.display = 'none'
        // }
        // else{
        //     const clickX = x
        //     const clickY = y_orig;
        //     displayCrossMark(clickX, clickY);
        // }
    } else if (event.key == 't' && metaState.brushSelection) {
        let [x, y] = performRayCasting()
        if (
            !(
                x < regionBounds[0] ||
                x > regionBounds[1] ||
                y < regionBounds[2] ||
                y > regionBounds[3]
            )
        ) {
            if (event.repeat) {
                var linePixels = []
                var x0 = lastX
                var y0 = lastY
                var x1 = x
                var y1 = y
                var steep: boolean = Math.abs(y1 - y0) > Math.abs(x1 - x0)
                if (steep) {
                    ;[x0, y0] = [y0, x0]
                    ;[x1, y1] = [y1, x1]
                }
                if (x0 > x1) {
                    ;[x0, x1] = [x1, x0]
                    ;[y0, y1] = [y1, y0]
                }
                var dx = x1 - x0
                var dy = y1 - y0
                var gradient
                if (dx == 0) {
                    gradient = 1
                } else {
                    gradient = dy / dx
                }
                var xend = x0
                var yend = y0
                var xpxl1 = xend
                var ypxl1 = yend
                if (steep) {
                    linePixels.push(ypxl1, xpxl1)
                    linePixels.push(ypxl1 + 1, xpxl1)
                } else {
                    linePixels.push(xpxl1, ypxl1)
                    linePixels.push(xpxl1, ypxl1 + 1)
                }
                var intery = yend + gradient
                xend = x1
                yend = y1
                var xpxl2 = xend
                var ypxl2 = yend
                if (steep) {
                    linePixels.push(ypxl2, xpxl2)
                    linePixels.push(ypxl2 + 1, xpxl2)
                } else {
                    linePixels.push(xpxl2, ypxl2)
                    linePixels.push(xpxl2, ypxl2 + 1)
                }
                if (steep) {
                    for (var z = xpxl1 + 1; z < xpxl2; z++) {
                        linePixels.push(Math.floor(intery), z)
                        linePixels.push(Math.floor(intery) + 1, z)
                        intery = intery + gradient
                    }
                } else {
                    for (var z = xpxl1 + 1; z < xpxl2; z++) {
                        linePixels.push(z, Math.floor(intery))
                        linePixels.push(z, Math.floor(intery) + 1)
                        intery = intery + gradient
                    }
                }
                brushLineHandler(linePixels, params.flood, params.clear)
            }
            lastX = x
            lastY = y
            brushHandler('t', x, regionDimensions[1] - 1 - y, params.flood, params.clear)
        }
    } else if (event.key == 'p' && metaState.polygonSelection) {
        let [x, y] = performRayCasting()
        y = regionDimensions[1] - 1 - y
        if (
            !(
                x < regionBounds[0] ||
                x > regionBounds[1] ||
                y < regionBounds[2] ||
                y > regionBounds[3]
            )
        ) {
            // y = regionDimensions[1] - y

            // const pixelIndex = y * regionDimensions[0] + x
            // const pixelVal = forestJson[pixelIndex]
            
            // if (pixelVal == 0){
            //     polygonSelectionHandler(x, y, params.flood, params.clear)
            // }
            polygonSelectionHandler(x, y, params.flood, params.clear)
        }
    } else if (event.key == 'o' && metaState.polygonSelection) {
        polygonFillHandler(params.flood, params.clear)
        // } else if (event.key == 's' && metaState.segEnabled) {
        //     let [x, y] = performRayCasting()
        //     segAnnotationHandler('s', x, y, params.flood, params.clear)
    } else if (event.key == 's' && metaState.segEnabled) {
        let [x, y] = performRayCasting()
        y = regionDimensions[1] - 1 - y
        connectedSegAnnotationHandler('s', x, y, params.flood, params.clear)
    }
    else if (event.key == ' ') {
        buttonPressHandlerSuperpixel()
    }
    else if (event.key == 'Enter') {
        buttonPressHandlerPrediction()
    }
}
const onKeyUp = (event: KeyboardEvent) => {
    if (event.key == 'g') {
        params.guide = 0
        uniforms.guide.value = params.guide
    }
    else if (event.key == ' '){
        params.superpixelKey = 0
        uniforms.superpixelKey.value = params.superpixelKey
    }
    else if (event.key == 'Enter'){
        params.predictionKey = 0
        uniforms.predictionKey.value = params.predictionKey
    }
}

async function startUp() {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('keydown', onKeyPress)
    window.addEventListener('keyup', onKeyUp)
    document.getElementById('cancel')?.addEventListener('click', () => {
        ;(document.getElementById('uploadForm') as HTMLFormElement).style.display = 'none'
        ;(document.getElementById('download') as HTMLElement).style.display = 'block'
    })
}

var diffuseTexture : THREE.Texture
var texContext : CanvasRenderingContext2D
;document.getElementById('submit')!.addEventListener('click', function(e) {
    let student_id = (document.getElementById('studentId') as HTMLInputElement).value
    let testRegion = (document.getElementById('testRegion') as HTMLInputElement).value
    
    if (student_id == ""){
        alert("Please provide your correct student id!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
        return
    }
    else if (testRegion == ""){
        alert("Please provide the correct Test Region ID!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
        return
    }

    e.preventDefault()
    if ((document.getElementById('stl') as HTMLInputElement).files![0]) {
        let file = (document.getElementById('stl') as HTMLInputElement).files![0]
        if (file.type == "image/png") {
            ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
            let fr = new FileReader()
            fr.onload = async function (e) {
                let image = document.createElement('img')
                image.src = e.target!.result as string
                image.onload = function() { 
                    regionBounds = [0, image.width, 0, image.height]
                    regionDimensions = [image.width, image.height]
                    controls.target = new THREE.Vector3(regionDimensions[0] / 2, regionDimensions[1] / 2, -2000)

                    var texCanvas = document.createElement('canvas')
                    texCanvas.width = image.width
                    texCanvas.height = image.height
                    texContext = texCanvas.getContext('2d')!

                    diffuseTexture = new THREE.Texture(texCanvas)

                    // var annCanvas = document.createElement('canvas')
                    annCanvas.width = image.width
                    annCanvas.height = image.height

                    // var predCanvas = document.createElement('canvas')
                    predCanvas.width = image.width
                    predCanvas.height = image.height
                    predContext = predCanvas.getContext('2d')!
                    
                    // var superpixelCanvas = document.createElement('canvas')
                    superpixelCanvas.width = image.width
                    superpixelCanvas.height = image.height
                    superpixelContext = superpixelCanvas.getContext('2d')!

                    // var confidenceCanvas = document.createElement('canvas')
                    confidenceCanvas.width = image.width
                    confidenceCanvas.height = image.height
                    confidenceContext = confidenceCanvas.getContext('2d')!

                    context = annCanvas.getContext('2d')!

                    annotationTexture = new THREE.Texture(annCanvas)

                    predictionTexture = new THREE.Texture(predCanvas) // 
                    superpixelTexture = new THREE.Texture(superpixelCanvas) // 
                    confidenceTexture = new THREE.Texture(confidenceCanvas) // 

                    uniforms.diffuseTexture.value = diffuseTexture
                    uniforms.annotationTexture.value = annotationTexture
                    uniforms.predictionTexture.value = predictionTexture // 
                    uniforms.superpixelTexture.value = superpixelTexture // 
                    uniforms.confidenceTexture.value = confidenceTexture // 
                    const meshMaterial = new THREE.RawShaderMaterial({
                        uniforms: uniforms,
                        vertexShader: terrainShader._VS,
                        fragmentShader: terrainShader._FS,
                    })
                    texContext.drawImage(image, 0, 0)

                    // if (!(document.getElementById("topology") as HTMLInputElement).checked) {
                    var imageData = texContext!.getImageData(0, 0, image.width, image.height).data
                    let temp = []
                    for (let i = 0; i < imageData.length; i+=4) {
                        temp.push(imageData[i])
                    }
                    data = {0: temp}
                    // }f

                    diffuseTexture.needsUpdate = true
                    annotationTexture.needsUpdate = true
                    // predictionTexture.needsUpdate = true // 
                    uniforms.dimensions.value = [image.width, image.height]
                    var formData = new FormData();
                    formData.append('file', file);
                    ajax({
                        url: `http://127.0.0.1:5000/stl?testRegion=${testRegion}`,
                        type: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        xhr: function() {
                            var xhr = new XMLHttpRequest()
                            xhr.responseType = 'blob'
                            return xhr
                        },
                        success: async function(data) {
                            const terrainLoader = new STLLoader()
                            try {
                                var test = window.URL.createObjectURL(data)
                                let response: THREE.BufferGeometry = await terrainLoader.loadAsync(
                                    test
                                ) 
                                mesh = new THREE.Mesh(response, meshMaterial)
                                mesh.receiveShadow = true
                                mesh.castShadow = true
                                mesh.position.set(0, 0, -100)
                                scene.add(mesh)

                                // // TODO: read output from backend and save somewhere to show on frontend
                                // const jsonUrl = 'http://127.0.0.1:5000/train'  // Replace with the actual URL
                                // const jsonData = await fetch(jsonUrl).then(response => response.json())

                                // // Process the JSON data (replace this part with your specific logic)
                                // console.log('JSON Data:', jsonData)
                                
                                var transformation_agg = 'avg';
                                if (uniforms.maxTransformation.value){
                                    transformation_agg = 'max'
                                }

                                var superpixel_agg = 'avg';
                                if (uniforms.minSuperpixel.value){
                                    superpixel_agg = 'min'
                                }

                                testRegion = encodeURIComponent(testRegion);
                                student_id = encodeURIComponent(student_id);

                                const superpixelBuffer = await fetch(`http://127.0.0.1:5000/superpixel?recommend=${1}
                                                                            &entropy=${uniforms.entropy.value}
                                                                            &probability=${uniforms.probability.value}
                                                                            &cod=${uniforms.cod.value}
                                                                            &transformation_agg=${transformation_agg}
                                                                            &superpixel_agg=${superpixel_agg}
                                                                            &taskId=${student_id}
                                                                            &testRegion=${testRegion}
                                                                            &initial=${1}
                                                                            &use_forest=${uniforms.use_forest.value}
                                                                            `).then(response => response.arrayBuffer());

                                console.log("superpixelBuffer: ", superpixelBuffer)

                                // Convert ArrayBuffer to base64
                                const base64ImageSuperpixel = arrayBufferToBase64(superpixelBuffer)

                                // Create an Image element
                                const imgSuperpixel = new Image();

                                // Set the source of the Image to the base64-encoded PNG data
                                imgSuperpixel.src = 'data:image/png;base64,' + base64ImageSuperpixel;

                                await new Promise(resolve => {
                                    imgSuperpixel.onload = resolve;
                                });

                                // Set canvas dimensions to match the image dimensions
                                superpixelCanvas.width = imgSuperpixel.width;
                                superpixelCanvas.height = imgSuperpixel.height;

                                console.log("height: ", superpixelCanvas.height)
                                console.log("width: ", superpixelCanvas.width)

                                // Draw the image on the canvas
                                superpixelContext!.drawImage(imgSuperpixel, 0, 0);
                                superpixelTexture.needsUpdate = true // 

                                const metrices_response = await fetch(`http://127.0.0.1:5000/metrics-json?taskId=${student_id}&testRegion=${testRegion}`);
                                metrices = await metrices_response.json();
                                console.log("metrices: ", metrices)


                                const predBuffer = await fetch(`http://127.0.0.1:5000/pred?taskId=${student_id}&testRegion=${testRegion}`).then(response => response.arrayBuffer());
                                console.log("arraybuffer: ", predBuffer)

                                // Convert ArrayBuffer to base64
                                const base64ImagePred = arrayBufferToBase64(predBuffer)

                                // Create an Image element
                                const imgPred = new Image();

                                // Set the source of the Image to the base64-encoded PNG data
                                imgPred.src = 'data:image/png;base64,' + base64ImagePred;

                                // Wait for the image to load
                                imgPred.onload = () => {

                                    // Set canvas dimensions to match the image dimensions
                                    predCanvas.width = imgPred.width;
                                    predCanvas.height = imgPred.height;

                                    console.log("height: ", predCanvas.height)
                                    console.log("width: ", predCanvas.width)

                                    // Draw the image on the canvas
                                    predContext!.drawImage(imgPred, 0, 0);
                                    predictionTexture.needsUpdate = true // 

                                    // // Add the canvas to the document or perform other actions
                                    // document.body.appendChild(predCanvas);
                                };
                                
                                // const forestResponse = await fetch(`http://127.0.0.1:5000/forest-json?testRegion=${testRegion}`);
                                // forestJson = await forestResponse.json();
                                // console.log("forestJson: ", forestJson)

                                // const gtResponse = await fetch(`http://127.0.0.1:5000/gt-json?testRegion=${testRegion}`);
                                // gtJson = await gtResponse.json();
                                // console.log("gtJson: ", gtJson)

                                const confidenceBuffer = await fetch(`http://127.0.0.1:5000/confidence?testRegion=${testRegion}`).then(response => response.arrayBuffer());
                                console.log("confidenceBuffer: ", confidenceBuffer)

                                // // forestArray = new DataView(confidenceBuffer);
                                // const pixelData = UPNG.decode(confidenceBuffer);
                                // forestArray = pixelData.data;

                                // // Iterate through pixels and store values in the dictionary
                                // for (let i = 0; i < forestArray.length; i += 3) {
                                //     // Calculate the pixel position (assuming a 3-byte RGBA format)
                                //     const pixelPosition = i / 3;
                                
                                //     // Store the pixel value in the dictionary
                                //     pixelDict[pixelPosition] = forestArray[i + 1]
                                // }

                                // console.log("index0: ", pixelDict[0])
                                // console.log("index1500: ", pixelDict[1500])
                                // console.log(forestArray)

                                // Convert ArrayBuffer to base64
                                const base64ImageConfidence = arrayBufferToBase64(confidenceBuffer)

                                // Create an Image element
                                const imgConfidence = new Image();

                                // Set the source of the Image to the base64-encoded PNG data
                                imgConfidence.src = 'data:image/png;base64,' + base64ImageConfidence;

                                await new Promise(resolve => {
                                    imgConfidence.onload = resolve;
                                });

                                // Set canvas dimensions to match the image dimensions
                                confidenceCanvas.width = imgConfidence.width;
                                confidenceCanvas.height = imgConfidence.height;

                                console.log("height: ", confidenceCanvas.height)
                                console.log("width: ", confidenceCanvas.width)

                                // Draw the image on the canvas
                                confidenceContext!.drawImage(imgConfidence, 0, 0);
                                confidenceTexture.needsUpdate = true // 

                                // confidenceJSON = await fetch('http://127.0.0.1:5000/confidence_json').then(response => response.json());
                                // console.log("confidenceJSON: ", confidenceJSON)

                                
                            } catch (e) {
                                console.error(`error on reading STL file a.stl`)
                            }                

                            // Get the container element
                            const jsonContainer = document.getElementById('metrices');
                            
                            if (jsonContainer){
                                console.log('jsonContainer found.');
                                // Create a <pre> element to display formatted JSON
                                const preElement = document.createElement('pre');

                                // Convert JSON object to a string with indentation
                                // const jsonString = JSON.stringify(metrices, null, 2);
                                // const jsonStringWithoutBraces = metrices.slice(1, -1);


                                // Set the content of the <pre> element to the formatted JSON string
                                preElement.textContent = metrices;

                                // Append the <pre> element to the container
                                jsonContainer.innerHTML = '';
                                jsonContainer.appendChild(preElement);
                                

                                ;(document.getElementById('metrices') as HTMLElement).style.display = 'block'
                            } else {
                                console.log('jsonContainer not found.');
                            }

                            ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
                            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
                        },
                        error: function(xhr, status, error) {
                            ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
                            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
                            console.log('Error uploading file: ' + error)
                        }
                    });
                }
            }
            fr.readAsDataURL(file)
        } else {
            alert('Invalid file type, must be .png!')
        }
    } else {
        alert('No data uploaded!')
    }

    // if ((document.getElementById('data') as HTMLInputElement).files![0] && (document.getElementById('topology') as HTMLInputElement).checked) {
    //     let file = (document.getElementById('data') as HTMLInputElement).files![0]
    //     if (file.type == "image/tiff") {
    //         ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
    //         ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    //         var formData = new FormData();
    //         formData.append('file', file);
    //         ajax({
    //             url: 'http://127.0.0.1:5000/topology',
    //             type: 'POST',
    //             data: formData,
    //             processData: false,
    //             contentType: false,
    //             dataType: 'json',
    //             success: async function(d) {
    //                 console.log(d)
    //                 data = d['data']
    //                 for (var i = 0; i < pers.length; i++) {
    //                     var thresh = pers[i]
    //                     persDatas[thresh] = new Int16Array(d['segmentation'][thresh])
    //                     var max = 0
    //                     var imageData = new Uint8Array(4 * persDatas[thresh].length)
    //                     // segsToPixels2[thresh] = {}
    //                     var imageData2 = new Uint8Array(4 * data[thresh].length)
    //                     for (var x = 0; x < regionDimensions[0]; x++) {
    //                         for (var y = 0; y < regionDimensions[1]; y++) {
    //                             var segID = persDatas[thresh][x + y * regionDimensions[0]]
    //                             if (segID > max) {
    //                                 max = segID
    //                             }
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4] = Math.floor(segID / 1000)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 1] = Math.floor((segID % 1000) / 100)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 2] = Math.floor((segID % 100) / 10)
    //                             imageData[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 3] = segID % 10
    //                             // if (segsToPixels2[thresh][segID]) {
    //                             //     segsToPixels2[thresh][segID].push(x)
    //                             // } else {
    //                             //     segsToPixels2[thresh][segID] = [x]
    //                             // }
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 1] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 2] = Math.floor(255 * data[thresh][y * regionDimensions[0] + x])
    //                             imageData2[(x + (regionDimensions[1] - y - 1) * regionDimensions[0]) * 4 + 3] = 255
    //                         }
    //                     }
    //                     segsMax[thresh] = max
    //                     persTextures[thresh] = new THREE.DataTexture(
    //                         imageData,
    //                         regionDimensions[0],
    //                         regionDimensions[1]
    //                     )
    //                     persTextures[thresh].needsUpdate = true
    //                     dataTextures[thresh] = new THREE.DataTexture(
    //                         imageData2,
    //                         regionDimensions[0],
    //                         regionDimensions[1]
    //                     )
    //                     dataTextures[thresh].needsUpdate = true
    //                 } 
    //                 uniforms.dataTexture.value = dataTextures[persVal]
    //                 uniforms.persTexture.value = persTextures[persVal]
    //                 uniforms.segsMax.value = segsMax[persVal]
    //                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
    //                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
    //             },
    //             error: function(xhr, status, error) {
    //                 console.log(xhr)
    //                 ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
    //                 ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
    //                 console.log('Error uploading file: ' + error)
    //             }
    //         });
    //     } else {
    //         alert('Invalid file type, must be .tiff for data!')
    //     }
    // } 

    if ((document.getElementById('texture') as HTMLInputElement).files![0]) {
        let file = (document.getElementById('texture') as HTMLInputElement).files![0]
        if (file.type == "image/png") {
            ;(document.getElementById('loader') as HTMLElement).style.display = 'block'
            ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
            let fr = new FileReader()
            fr.onload = async function (e) {
                let image = document.createElement('img')
                image.src = e.target!.result as string
                image.onload = function() { 
                    texContext!.drawImage(image, 0, 0)
                }
            }
            fr.readAsDataURL(file)
        } else {
            alert('Invalid file type, must be .png!')
        }
    }
})



function arrayBufferToBase64(buffer: ArrayBuffer): string {
    // const binary = String.fromCharCode(...new Uint8Array(buffer));
    // return window.btoa(binary);

    // const binary = Buffer.from(buffer).toString('base64');
    // return binary;

    var base64 = btoa(
        new Uint8Array(buffer)
          .reduce((data, byte) => data + String.fromCharCode(byte), '')
      );
    return base64;
}


function disposeUniform() {
    type ObjectKeyUniforms = keyof typeof uniforms
    for (let key in uniforms) {
        if (uniforms[key as ObjectKeyUniforms]) {
            let x: any = uniforms[key as ObjectKeyUniforms]
            if (x['type'] !== undefined && x['type'] == 't') {
                x['value'].dispose()
                uniforms[key as ObjectKeyUniforms].value = new THREE.Texture()
            }
        }
    }
    for (let key in persTextures) {
        // persTextures[key].dispose()
        persTextures[key] = new THREE.Texture()
        dataTextures[key] = new THREE.Texture()
    }
}

window.addEventListener('resize', onWindowResize, false)
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    gui.width = window.innerWidth / 5
    render()
}

function animate() {
    requestAnimationFrame(animate)
    stats_mb.update()
    if (camera.position.z <= 100) {
        camera.position.z = 100
        camera.updateProjectionMatrix()
    }
    if (!overRideControl) {
        controls.update()
    }
    TWEEN.update()
    // let position = new THREE.Vector3()
    // camera.getWorldPosition(position)
    render()
}

function render() {
    renderer.render(scene, camera)
}

function getCameraLastStage() {
    return {
        position: camera.position.clone(),
        lookAt: controls.target,
    }
}

animate()

export {
    canvas,
    startUp,
    controls,
    mesh,
    pointer,
    renderer,
    camera,
    TWEEN,
    raycaster,
    scene,
    params,
    uniforms,
    gui,
    disposeUniform,
    annCanvas,
    arrayBufferToBase64,
    superpixelContext,
    superpixelTexture,
    superpixelCanvas,
    predCanvas,
    predictionTexture,
    predContext,
    confidenceCanvas,
    confidenceTexture,
    confidenceContext,
    context, // for annotation
    annotationTexture,
}
