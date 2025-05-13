import { update } from '@tweenjs/tween.js'
import { Camera, Raycaster, Scene } from 'three'
import { MapControls } from 'three/examples/jsm/controls/OrbitControls'
import {
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
    context,
    annotationTexture
} from './client'
import { terrainDimensions } from './constants'
import * as JSZip from 'jszip'
import * as THREE from 'three'
import { time } from 'console'
// import * as fs from 'fs'

var zip: JSZip = new JSZip()
const pixelCount = 7617024
let button1: HTMLButtonElement, button2: HTMLButtonElement
let button3: HTMLButtonElement, button4: HTMLButtonElement
type ObjectKeyParams = keyof typeof params
type ObjectKeyUniforms = keyof typeof uniforms

interface sessionDataType {
    name: string
    testRegion: string
    sessionStart: Date | null
    sessionEnd: Date | null
    'totalSessionTime_M:S:MS': string
    wasCompleted: boolean
    annotatedPixelCount: number
    numberofClick: number
    numberofUndo: number
    numberofRedo: number
    metaState: Object
}

interface annotationTimeType {
    [key: number]: number
}

let annotationTimeTable: annotationTimeType = {}

interface gameEventType {
    label: string
    flood: boolean
    clear: boolean
    clickPosition?: THREE.Vector2
    keyPressed?: string
    brushSize?: number
    x?: number
    y?: number
    linePoints?: Array<number>
    undone?: boolean
    redone?: boolean
    persistanceThreshold?: number
    aspectRatio: number
    cameraPosition: THREE.Vector3
    targetPosition: THREE.Vector3
    time: Date
    annotatedPixelCount?: number
}

interface gameStateType {
    [key: string]: gameEventType
}

let metaState = {
    BFS: true,
    brushSelection: true,
    polygonSelection: true,
    segEnabled: true,
    region: '1',
    quadrant: 0,
    flat: 0,
}
if (window.location.hash) {
    if (window.location.hash.search('region') != -1) {
        metaState.region = window.location.hash[window.location.hash.search('region') + 7]
        if (metaState.region == '4') {
            alert('Region 4 not supported!')
        }
    }
    if (window.location.hash.search('BFS') != -1) {
        var bfs = window.location.hash[window.location.hash.search('BFS') + 4]
        if (bfs == '0') {
            document.getElementById('BFS')!.style.display = 'none'
            document.getElementById('polygonSelection')!.style.display = 'none'
            document.getElementById('polygonSelection2')!.style.display = 'none'
            document.getElementById('menuBFS')!.style.display = 'none'
            metaState.BFS = false
            metaState.polygonSelection = false
        }
    }
    if (window.location.hash.search('segmentation') != -1) {
        var seg = window.location.hash[window.location.hash.search('segmentation') + 13]
        if (seg == '0') {
            document.getElementById('segEnabled')!.style.display = 'none'
            document.getElementById('segEnabled2')!.style.display = 'none'
            document.getElementById('menuSegmentation')!.style.display = 'none'
            metaState.segEnabled = false
        }
    }
    if (window.location.hash.search('quadrant') != -1) {
        metaState.quadrant = parseInt(
            window.location.hash[window.location.hash.search('quadrant') + 9]
        )
    }
    if (window.location.hash.search('flat') != -1) {
        metaState.flat = parseInt(window.location.hash[window.location.hash.search('flat') + 5])
    }
}
const regionDimensions = terrainDimensions[metaState.region]
console.log("regionDimensions: ", regionDimensions)
let regionBounds = [0, regionDimensions[0], 0, regionDimensions[1]]
if (metaState.quadrant == 1) {
    regionBounds[1] = Math.floor(regionDimensions[0] / 2)
    regionBounds[2] = Math.ceil(regionDimensions[1] / 2)
} else if (metaState.quadrant == 2) {
    regionBounds[0] = Math.ceil(regionDimensions[0] / 2)
    regionBounds[2] = Math.ceil(regionDimensions[1] / 2)
} else if (metaState.quadrant == 3) {
    regionBounds[1] = Math.floor(regionDimensions[0] / 2)
    regionBounds[3] = Math.floor(regionDimensions[1] / 2)
} else if (metaState.quadrant == 4) {
    regionBounds[0] = Math.ceil(regionDimensions[0] / 2)
    regionBounds[3] = Math.floor(regionDimensions[1] / 2)
}

const sessionData: sessionDataType = {
    name: 'anonymous',
    testRegion: '1',
    sessionStart: null,
    sessionEnd: null,
    'totalSessionTime_M:S:MS': '0:0:0',
    wasCompleted: false,
    annotatedPixelCount: 0,
    numberofClick: 0,
    numberofUndo: 0,
    numberofRedo: 0,
    metaState: metaState,
}

// vector.applyMatrix(camera.matrixWorld)

const gameState: Array<gameStateType> = []

async function readstateFile() {
    let _fetchData: any
    const response = await fetch('./data/session_.json')
    _fetchData = await response.json()
    return _fetchData
}

// async function readMetaFile() {
//     let _fetchData: any
//     const response = await fetch('./data/meta_session_.json')
//     _fetchData = await response.json()
//     return _fetchData
// }

// ;(async () => {
//     let result = await readMetaFile()
// })()

function logMyState(
    key: string,
    event: string,
    flood: boolean,
    clear: boolean,
    camera: THREE.PerspectiveCamera,
    pointer?: THREE.Vector2,
    x?: number,
    y?: number,
    brushSize?: number,
    linePoints?: Array<number>,
    time?: Date
) {
    let tempS: string = event

    let stateData
    if (brushSize != undefined) {
        stateData = {
            label: tempS,
            flood: flood,
            clear: clear,
            clickPosition: pointer,
            keyPressed: key,
            x: x,
            y: y,
            aspectRatio: camera.aspect,
            cameraPosition: camera.position.clone(),
            targetPosition: controls.target.clone(),
            time: time ? time : new Date(),
            annotatedPixelCount: sessionData.annotatedPixelCount,
        }
    }

    if (linePoints != undefined) {
        stateData = {
            label: tempS,
            flood: flood,
            clear: clear,
            aspectRatio: camera.aspect,
            keyPressed: key,
            cameraPosition: camera.position.clone(),
            targetPosition: controls.target.clone(),
            time: time ? time : new Date(),
            linePoints: linePoints,
            annotatedPixelCount: sessionData.annotatedPixelCount,
        }
    } else {
        stateData = {
            label: tempS,
            flood: flood,
            clear: clear,
            clickPosition: pointer,
            keyPressed: key,
            x: x,
            y: y,
            aspectRatio: camera.aspect,
            cameraPosition: camera.position.clone(),
            targetPosition: controls.target.clone(),
            time: time ? time : new Date(),
            brushSize: brushSize,
            persistanceThreshold: params.pers,
            annotatedPixelCount: sessionData.annotatedPixelCount,
        }
    }
    gameState.push({ mouseEvent: stateData })
}

function checkpoint() {
    const text = JSON.stringify(gameState)
    const filename = 'checkpoint_' + sessionData.name + '.json'
    var element = document.createElement('a')
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text))
    element.setAttribute('download', filename)
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
}

function convertToSecMins(millisecond: number) {
    const minutes = Math.floor(millisecond / 60000)
    const seconds = ((millisecond % 60000) / 1000).toFixed(0)
    const milliseconds = (millisecond % 1000).toFixed(0)
    return minutes + ':' + (+seconds < 10 ? '0' : '') + seconds + ':' + milliseconds
}

function resetCamera(controls: any) {
    controls.reset()
    return controls
}

// function startSession() {
//     // event.preventDefault()
//     // ;(event.target as HTMLButtonElement).style.display = 'none'
//     let startStateData = {
//         label: 'start',
//         aspectRatio: camera.aspect,
//         cameraPosition: camera.position.clone(),
//         targetPosition: controls.target.clone(),
//         time: new Date(),
//         flood: true,
//         clear: false,
//     }
//     gameState.push({ start: startStateData })
//     if (!sessionData.sessionStart) {
//         sessionData.sessionStart = new Date()
//     }
//     new TWEEN.Tween(controls.target)
//         .to(
//             {
//                 x: 0,
//                 y: 0,
//                 z: 0,
//             },
//             1000
//         )
//         .easing(TWEEN.Easing.Cubic.Out)
//         .onUpdate(() => {
//             controls.update()
//         })
//         .start()

//     new TWEEN.Tween(camera.position)
//         .to(
//             {
//                 x: 0,
//                 y: 0,
//                 z: 1000,
//             },
//             1000
//         )
//         .easing(TWEEN.Easing.Cubic.Out)
//         .onUpdate(() => {
//             camera.updateProjectionMatrix()
//         })
//         .start()
//     startUp()
// }

function startSession() {
    // event.preventDefault()
    // ;(event.target as HTMLButtonElement).style.display = 'none'
    let startStateData = {
        label: 'start',
        aspectRatio: camera.aspect,
        cameraPosition: camera.position.clone(),
        targetPosition: controls.target.clone(),
        time: new Date(),
        flood: true,
        clear: false,
    }
    gameState.push({ start: startStateData })
    if (!sessionData.sessionStart) {
        sessionData.sessionStart = new Date()
    }
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
                z: 1000,
            },
            1000
        )
        .easing(TWEEN.Easing.Cubic.Out)
        .onUpdate(() => {
            camera.updateProjectionMatrix()
        })
        .start()
    startUp()
}

function endSession(event: Event) {
    event.preventDefault()
    ;(event.target as HTMLButtonElement).style.display = 'none'
    sessionData.sessionEnd = new Date()
    sessionData.wasCompleted = false
    let totalSessionTime = Math.abs(
        sessionData.sessionStart!.valueOf() - sessionData.sessionEnd!.valueOf()
    )
    sessionData['totalSessionTime_M:S:MS'] = convertToSecMins(totalSessionTime)
    if (sessionData.annotatedPixelCount > 0.75 * pixelCount) {
        sessionData.wasCompleted = true
    }
}

function downloadSession(event: Event) {
    disposeUniform()
    const _data = JSON.stringify(gameState)
    const _fileName = 'session_' + sessionData.name + '.json'
    zip.file(_fileName, _data)
    // download(_fileName, _data)
    const _metadata = JSON.stringify(sessionData)
    const _metaFileName = 'meta_session_' + sessionData.name + '.json'
    // download(_metaFileName, _metadata)
    zip.file(_metaFileName, _metadata)
    let imageName = 'annotatedImg.png'
    if (sessionData.name) {
        imageName = 'annotatedImg_' + sessionData.name + '.png'
    }

    var url = annCanvas.toDataURL()
    var index = url.indexOf(',')
    if (index !== -1) {
        url = url.substring(index + 1, url.length)
    }
    zip.file(imageName, url, { base64: true })

    let imageNamePred = 'AL_prediction.png'
    if (sessionData.name) {
        imageNamePred = 'AL_prediction_' + sessionData.name + '.png'
    }

    var urlPred = predCanvas.toDataURL()
    var indexPred = urlPred.indexOf(',')
    if (indexPred !== -1) {
        urlPred = urlPred.substring(indexPred + 1, urlPred.length)
    }
    zip.file(imageNamePred, urlPred, { base64: true })

    zip.generateAsync({
        type: 'base64',
    }).then(function (content) {
        var link = document.createElement('a')
        link.href = 'data:application/zip;base64,' + content
        if (sessionData.name == null || sessionData.name == '') {
            sessionData.name = 'anonymous'
        }
        link.download = sessionData.name + '-annotation'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    })
    var link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('target', '_blank')
    link.setAttribute('download', imageName)
    // link.click()
    ;(document.getElementById('uploadForm') as HTMLFormElement).style.display = 'block'
    disposeHierarchy(scene, disposeNode)
    renderer.renderLists.dispose()
    disposeNode(scene)
}

function downloadPredictionSession(event: Event) {
    disposeUniform()
    // const _data = JSON.stringify(gameState)
    // const _fileName = 'session_' + sessionData.name + '.json'
    // zip.file(_fileName, _data)
    // // download(_fileName, _data)
    // const _metadata = JSON.stringify(sessionData)
    // const _metaFileName = 'meta_session_' + sessionData.name + '.json'
    // // download(_metaFileName, _metadata)
    // zip.file(_metaFileName, _metadata)
    let imageName = 'AL_prediction.png'
    if (sessionData.name) {
        imageName = 'AL_prediction_' + sessionData.name + '.png'
    }

    var url = predCanvas.toDataURL()
    var index = url.indexOf(',')
    if (index !== -1) {
        url = url.substring(index + 1, url.length)
    }
    zip.file(imageName, url, { base64: true })
    zip.generateAsync({
        type: 'base64',
    }).then(function (content) {
        var link = document.createElement('a')
        link.href = 'data:application/zip;base64,' + content
        if (sessionData.name == null || sessionData.name == '') {
            sessionData.name = 'anonymous'
        }
        link.download = sessionData.name + '-annotation'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    })
    var link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('target', '_blank')
    link.setAttribute('download', imageName)
    // link.click()
    ;(document.getElementById('uploadForm') as HTMLFormElement).style.display = 'block'
    disposeHierarchy(scene, disposeNode)
    renderer.renderLists.dispose()
    disposeNode(scene)
}

function dataURItoBlob(dataURI: string) {
    // convert base64 to raw binary data held in a string
    // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
    var byteString = atob(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    //Old Code
    //write the ArrayBuffer to a blob, and you're done
    //var bb = new BlobBuilder();
    //bb.append(ab);
    //return bb.getBlob(mimeString);

    //New Code
    return new Blob([ab], {type: mimeString});
}


function showLoadingScreen(){
    ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'block'
    // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'block'
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    ;(document.getElementById('metrices') as HTMLElement).style.display = 'none'
}

function hideLoadingScreen(){
    ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
    // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
}

// Function to poll the backend
async function pollBackendTask(taskId: string) {
    const response = await fetch(`http://127.0.0.1:5000/check-status?taskId=${taskId}&testRegion=${sessionData.testRegion}`);
    const data = await response.json();

    // console.log("data: ", data)
  
    if (data.status === 'completed') {
        // Backend task is completed, handle the response
        console.log('Backend task completed:', data.result);

        // context!.clearRect(0, 0, annCanvas.width, annCanvas.height);
        // annotationTexture.needsUpdate = true;
  
        // Continue with other actions on the frontend
        const superpixelBuffer = await fetch(`http://127.0.0.1:5000/superpixel?recommend=${0}&use_forest=${uniforms.use_forest.value}`).then(response => response.arrayBuffer());
        // console.log("superpixelBuffer: ", superpixelBuffer)

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

        // console.log("height: ", superpixelCanvas.height)
        // console.log("width: ", superpixelCanvas.width)

        // Draw the image on the canvas
        superpixelContext!.drawImage(imgSuperpixel, 0, 0);
        superpixelTexture.needsUpdate = true // 

        const predBuffer = await fetch(`http://127.0.0.1:5000/pred?taskId=${sessionData.name}&testRegion=${sessionData.testRegion}`).then(response => response.arrayBuffer());
        // console.log("arraybuffer: ", predBuffer)

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

            // console.log("height: ", predCanvas.height)
            // console.log("width: ", predCanvas.width)

            // Draw the image on the canvas
            predContext!.drawImage(imgPred, 0, 0);
            predictionTexture.needsUpdate = true // 
        };

        ;(document.getElementById('exploration') as HTMLElement).style.display = 'block'
        ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
        // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    
        // Hide the loading screen
        //   hideLoadingScreen();
    } else {
      // Backend task is still in progress, continue polling
      setTimeout(() => pollBackendTask(taskId), 120000); // Poll every 2 mins
    }
  }

// 
async function retrainSession(event: Event) {
    event.stopPropagation();
    
    // TODO: check what this does???
    // disposeUniform()
    console.log("Retrain session")

    var dataURL = annCanvas.toDataURL()

    // Create a FormData object and append the image data
    var formData = new FormData();
    const dataURLFile = dataURItoBlob(dataURL);
    formData.append('image', dataURLFile);

    const taskId = encodeURIComponent(sessionData.name);
    const testRegion = encodeURIComponent(sessionData.testRegion);

    // Send a POST request to the Flask backend

    var transformation_agg = 'avg';
    if (uniforms.minTransformation.value){
        transformation_agg = 'min'
    }
    else if (uniforms.maxTransformation.value){
        transformation_agg = 'max'
    }

    var superpixel_agg = 'avg';
    if (uniforms.minSuperpixel.value){
        superpixel_agg = 'min'
    }
    else if (uniforms.maxSuperpixel.value){
        superpixel_agg = 'max'
    }

    showLoadingScreen();

    const response = await fetch(`http://127.0.0.1:5000/retrain?taskId=${taskId}
                                    &entropy=${uniforms.entropy.value}
                                    &probability=${uniforms.probability.value}
                                    &cod=${uniforms.cod.value}
                                    &transformation_agg=${transformation_agg}
                                    &superpixel_agg=${superpixel_agg}
                                    &sc_loss=${uniforms.sc_loss.value}
                                    &cod_loss=${uniforms.cod_loss.value}
                                    &use_forest=${uniforms.use_forest.value}
                                    &testRegion=${testRegion}`, {
                        method: 'POST',
                        body: formData,
                    });

    const data = await response.json();

    // Check if the task was successfully started
    if (data.status === 'success') {
        // Start polling the backend for task status
        // pollBackendTask(data.taskId);
        

        // Continue with other actions on the frontend
        const superpixelBuffer = await fetch(`http://127.0.0.1:5000/superpixel?recommend=${0}
                                                                                &taskId=${taskId}
                                                                                &testRegion=${testRegion}
                                                                                &use_forest=${uniforms.use_forest.value}
                                                                            `).then(response => response.arrayBuffer());
        // console.log("superpixelBuffer: ", superpixelBuffer)

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

        // console.log("height: ", superpixelCanvas.height)
        // console.log("width: ", superpixelCanvas.width)

        // Draw the image on the canvas
        superpixelContext!.drawImage(imgSuperpixel, 0, 0);
        superpixelTexture.needsUpdate = true // 

        const metrices_response = await fetch(`http://127.0.0.1:5000/metrics-json?taskId=${taskId}&testRegion=${testRegion}`);
        const metrices = await metrices_response.json();
        console.log("metrices: ", metrices)

        const predBuffer = await fetch(`http://127.0.0.1:5000/pred?taskId=${taskId}&testRegion=${testRegion}`).then(response => response.arrayBuffer());
        // console.log("arraybuffer: ", predBuffer)

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

            // console.log("height: ", predCanvas.height)
            // console.log("width: ", predCanvas.width)

            // Draw the image on the canvas
            predContext!.drawImage(imgPred, 0, 0);
            predictionTexture.needsUpdate = true // 
        };

        // Get the container element
        const jsonContainer = document.getElementById('metrices');
                            
        if (jsonContainer){
            console.log('jsonContainer found.');
            // Create a <pre> element to display formatted JSON
            const preElement = document.createElement('pre');

            // // Convert JSON object to a string with indentation
            // const jsonString = JSON.stringify(metrices, null, 2);
            // const jsonStringWithoutBraces = jsonString.slice(1, -1);

            // Set the content of the <pre> element to the formatted JSON string
            preElement.textContent = metrices;

            // Append the <pre> element to the container
            jsonContainer.innerHTML = '';
            jsonContainer.appendChild(preElement);

            ;(document.getElementById('metrices') as HTMLElement).style.display = 'block'
        } else {
            console.log('jsonContainer not found.');
        }

        ;(document.getElementById('exploration') as HTMLElement).style.display = 'block'
        ;(document.getElementById('loaderSide') as HTMLElement).style.display = 'none'
        // ;(document.getElementById('loaderTrain') as HTMLElement).style.display = 'none'
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'

    } else {
        // Handle the case where the task couldn't be started
        console.error('Failed to start backend task');
        hideLoadingScreen();
    }
}

function hideModal() {
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'none'
    ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'block'
    let userId = (document.getElementById('studentId') as HTMLInputElement).value
    let testRegion = (document.getElementById('testRegion') as HTMLInputElement).value
    console.log("testRegion: ", testRegion)
    console.log("userId: ", userId)

    if (userId == ""){
        alert("Please provide your correct student id!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
    }
    else if (testRegion == ""){
        alert("Please provide the correct Test Region ID!")
        ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
        ;(document.getElementById('ui-menu') as HTMLElement).style.display = 'none'
    }
    else{
        sessionData.name = userId
        sessionData.testRegion = testRegion
    }

    console.log("testRegion: ", sessionData.testRegion)
    console.log("userId: ", sessionData.name)

    startSession()
}

function getLocalCordinate(_cordiante: THREE.Vector3) {
    mesh!.updateMatrixWorld()
    const localPoint = mesh!.worldToLocal(_cordiante)
    return localPoint
}

function doubleClickHandler(event: MouseEvent) {
    event.preventDefault()
    let ndcX = (event.clientX / renderer.domElement.clientWidth) * 2 - 1
    let ndcY = -(event.clientY / renderer.domElement.clientHeight) * 2 + 1
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, camera)

    const intersection = raycaster.intersectObjects(scene.children, true)
    if (intersection.length > 0) {
        const point = intersection[0].point //this is not local cordinate point rather world cordinate
        new TWEEN.Tween(controls.target)
            .to(
                {
                    x: point.x,
                    y: point.y,
                    z: point.z,
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
                    x: point.x,
                    y: point.y,
                    z: camera.position.z,
                },
                1000
            )
            .easing(TWEEN.Easing.Cubic.Out)
            .onUpdate(() => {
                camera.updateProjectionMatrix()
            })
            .start()
    }
}

function toggleAnnoation() {
    var ul_button = document.createElement('ul')
    var li = document.createElement('li')
    li.classList.add('customList', 'outList')
    // let span = document.createElement('span')
    // span.classList.add('property-name')
    // span.innerHTML = 'Annotate'
    // li.appendChild(span)
    let div = document.createElement('div')
    div.classList.add('btn-group', 'btn-group-toggle')
    button1 = document.createElement('button')
    button1.classList.add('ci', 'btn', 'active')
    button1.setAttribute('data-myid', 'flood')
    button1.innerHTML = 'FLOOD'
    button2 = document.createElement('button')
    button2.classList.add('ci', 'btn')
    button2.setAttribute('data-myid', 'dry')
    button2.innerHTML = 'DRY'
    div.appendChild(button1)
    div.appendChild(button2)
    li.appendChild(div)
    button1.addEventListener('click', setActiveButton)
    button2.addEventListener('click', setActiveButton)
    ul_button.appendChild(li)
    // document.body.appendChild(li)
    var li2 = document.createElement('li')
    li2.classList.add('customList', 'outList2')
    let div2 = document.createElement('div')
    div2.classList.add('btn-group', 'btn-group-toggle')
    button3 = document.createElement('button')
    button3.classList.add('ci', 'btn', 'active')
    button3.setAttribute('data-myid', 'fill')
    button3.innerHTML = 'FILL'
    button4 = document.createElement('button')
    button4.classList.add('ci', 'btn')
    button4.setAttribute('data-myid', 'clear')
    button4.innerHTML = 'ERASE'
    div2.appendChild(button3)
    div2.appendChild(button4)
    li2.appendChild(div2)
    button3.addEventListener('click', setActiveButton2)
    button4.addEventListener('click', setActiveButton2)
    ul_button.appendChild(li2)
    document.body.appendChild(ul_button)
}

function updateUniform(input: any) {
    input.forEach((element: any, index: number) => {
        uniforms[element as ObjectKeyUniforms].value = params[element as ObjectKeyParams] as any
    })
}

function setActiveButton(event: MouseEvent) {
    event.preventDefault()
    button1.classList.remove('active')
    button2.classList.remove('active')
    ;(event.target as HTMLButtonElement).classList.add('active')
    type ObjectKeyParams = keyof typeof params
    let myId = (event.target as HTMLButtonElement).dataset.myid as ObjectKeyParams
    params['dry'] = false
    params['flood'] = false
    // params[myId] = true
    if (myId == 'flood') {
        params['flood'] = true
    } else {
        params['dry'] = true
    }
    updateUniform(['dry', 'flood'])
}

function setActiveButton2(event: MouseEvent) {
    event.preventDefault()
    button3.classList.remove('active')
    button4.classList.remove('active')
    ;(event.target as HTMLButtonElement).classList.add('active')
    type ObjectKeyParams = keyof typeof params
    let myId = (event.target as HTMLButtonElement).dataset.myid as ObjectKeyParams
    if (myId == 'clear') {
        params['clear'] = true
    } else {
        params['clear'] = false
    }
}

function init() {
    // document.getElementById('start')?.addEventListener('click', startSession)
    document.getElementById('end')?.addEventListener('click', endSession)
    document.getElementById('download')?.addEventListener('click', downloadSession)
    document.getElementById('checkpoint')?.addEventListener('click', checkpoint)
    document.getElementById('exploration')?.addEventListener('click', hideModal)
    document.getElementById('retrain')?.addEventListener('click', retrainSession) // 
    document.getElementById('download_pred')?.addEventListener('click', downloadPredictionSession) // 
    toggleAnnoation()
    renderer.domElement.addEventListener('dblclick', doubleClickHandler, true)
    // renderer.domElement.addEventListener('click', hideSetting, false)
}

function hideSetting() {
    // const dgDiv = document.querySelector('div.main div') as HTMLElement
    // dgDiv.style.height = '101px'
    // const dgUl = document.querySelector('li.folder div.dg ul') as HTMLUListElement
    // dgUl.classList.add('closed')
    gui.close()
}

function initVis() {
    ;(document.getElementById('loader') as HTMLElement).style.display = 'none'
    ;(document.getElementById('modal-wrapper') as HTMLElement).style.display = 'block'
}

function makeContinousData(data: any, gap: number) {
    let timeIndex: Array<any> = Object.keys(data)
    let lastIndex = Math.ceil(+timeIndex[timeIndex.length - 1] / 3)
    let result = new Array(lastIndex).fill(0)
    for (let key in data) {
        let key1 = Math.floor(+key / gap)
        result[key1] += +data[key]
    }
    convertArrayIntoCSV(result)
}

function convertArrayIntoCSV(data: any) {
    const fields = ['time', 'pixel_counts']
    let result = fields.join(',') + '\n'

    data.forEach((element: number, index: number) => {
        let row = [index, element]
        result += row.join(',') + '\n'
    })

    downloadCSV(result, '3sGap')
}

function convertToCSV(timedata: any) {
    const fields = ['time', 'pixel_counts']
    let result = fields.join(',') + '\n'
    for (let key in timedata) {
        const localData = [key, timedata[key]]
        result += localData.join(',') + '\n'
    }
    downloadCSV(result, 'eventTime')
}

function downloadCSV(csv_data: any, name: string) {
    var hiddenElement = document.createElement('a')
    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv_data)
    hiddenElement.target = '_blank'
    hiddenElement.download = name + '.csv'
    hiddenElement.click()
}

function disposeNode(node: any) {
    if (node instanceof THREE.Mesh) {
        if (node.geometry) {
            node.geometry.dispose()
            node.geometry = undefined
        }

        if (node.material) {
            for (let key in node.material.uniforms) {
                if (
                    key == 'annotationTexture' ||
                    key == 'colormap' ||
                    key == 'diffuseTexture' ||
                    key == 'persTexture' ||
                    key == 'predictionTexture' || // 
                    key == 'superpixelTexture' || // 
                    key == 'confidenceTexture' // 
                ) {
                    node.material.uniforms[key].value.dispose()
                }
            }
            node.material.dispose()
            node.material = undefined
        }

        scene.remove(node)
        node = undefined
    }
}

function disposeHierarchy(node: any, callback: any) {
    for (var i = node.children.length - 1; i >= 0; i--) {
        var child = node.children[i]
        disposeHierarchy(child, callback)
        callback(child)
    }
}

export {
    metaState,
    regionBounds,
    regionDimensions,
    resetCamera,
    startSession,
    endSession,
    init,
    initVis,
    sessionData,
    gameState,
    logMyState,
    getLocalCordinate,
    readstateFile,
    // readMetaFile,
    toggleAnnoation,
    annotationTimeTable,
    makeContinousData,
    convertToCSV,
    disposeNode,
}
