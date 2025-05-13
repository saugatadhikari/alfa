const _VS = `#version 300 es
precision highp float;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;

out vec3 vNormal;
out vec3 vPosition;

#define saturate(a) clamp( a, 0.0, 1.0 )

void main(){
    vNormal = normal;
    vPosition = position.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    // vPosition = gl_Position.xyz*vec3(0.5,0.5,0.5) + vec3(0.5,0.5,0.5);
  }
`

const _FS = `#version 300 es

precision mediump sampler2DArray;
precision highp float;
precision highp int;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform vec3 cameraPosition;
uniform sampler2D diffuseTexture;
uniform sampler2D dataTexture;
uniform sampler2D annotationTexture;
uniform sampler2D predictionTexture;
uniform sampler2D superpixelTexture;
uniform sampler2D confidenceTexture;
uniform sampler2D persTexture;
uniform sampler2D colormap;
uniform int annotation;
uniform int prediction;
uniform int superpixel;
uniform int confidence;
uniform int persShow;
uniform int data;
uniform float hoverValue;
uniform float segsMax;
uniform int guide;
uniform bool flood;
uniform bool dry;
uniform int z;
uniform vec2 dimensions;
uniform int quadrant;
uniform int superpixelKey;
uniform int predictionKey;

in vec3 vNormal;
in vec3 vPosition;

out vec4 out_FragColor;

vec4 sampleTexture(sampler2D sampleTex, vec2 coords) {
  return texture(sampleTex, coords / dimensions);
}

void main(){
    if (quadrant > 2 && vPosition.y > float(dimensions.y) / 2.0) {
        discard;
    }
    if ((quadrant == 1 || quadrant == 3) && vPosition.x > float(dimensions.x) / 2.0) {
        discard;
    }
    if ((quadrant == 1 || quadrant == 2) && vPosition.y < float(dimensions.y) / 2.0) {
        discard;
    }    
    if ((quadrant == 2 || quadrant == 4) && vPosition.x < float(dimensions.x) / 2.0) {
        discard;
    }
    vec3 color;
    if (data > 0) {
        color = sampleTexture(dataTexture, vPosition.xy).rgb;
    } else {
        color = sampleTexture(diffuseTexture, vPosition.xy).rgb;
    }
    vec4 _segID = sampleTexture(persTexture, vPosition.xy).rgba * 255.0;
      float segID = _segID.x*1000.0 + _segID.y*100.0 + _segID.z*10.0 + _segID.w*1.0;
      if(guide > 0){
        if(abs(segID - hoverValue) < 0.1){
          if(flood){
            color = 0.5*color + 0.5*vec3(1, 0, 0);
          }
          if(dry){
            color = 0.5*color + 0.5*vec3(0, 0, 1.0);
          }
        }
      }
      else{
        if (persShow > 0) {
        
            if (persShow == 1 || persShow == 3) {
              color = 0.9 * color + 0.1 * texture(colormap, vec2(segID / segsMax, 0)).rgb;
            }
            if (data < 1) { color = texture(colormap, vec2(segID / segsMax, 0)).rgb; }
    
            if (persShow > 1) {
              const vec2 neighbors[12] = vec2[12](
                // Diagonal neighbors
                vec2(-1, -1), vec2(1, -1), vec2(1, 1),vec2(-1, 1), 
                // Cross neighbors
                vec2(-1, 0), vec2(0, -1), vec2(0, 1), vec2(1, 0),
                // Further cross neighbors
                vec2(-2, 0), vec2(0, -2), vec2(0, 2), vec2(2, 0)
              );
              for (int i = 0; i < 8; i++) {
                vec4 _segIDn = sampleTexture(persTexture, vPosition.xy + neighbors[i]).rgba * 255.0;
                float segIDn = _segIDn.x*1000.0 + _segIDn.y*100.0 + _segIDn.z*10.0 + _segIDn.w*1.0;
                if (abs(segIDn - segID) > 0.001) {
                    if (data > 0) {
                        color = vec3(1.0, 0.0, 0.0);
                    }
                }
              }
            }    
          }
       }


    // if (annotation == 1) {
    //     vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
    //     out_FragColor = vec4(color + aColor, 1.0);
    // } else {
    //     out_FragColor = vec4(color, 1.0);
    // }

    if (prediction == 1 || predictionKey == 1){
      vec3 pColor = sampleTexture(predictionTexture, vPosition.xy).rgb;
      
      if (annotation == 1){
        vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
        out_FragColor = vec4(color + aColor + pColor, 1.0);
      } else {
        out_FragColor = vec4(color + pColor, 1.0);
      }
    }
    else if (superpixel == 1 || superpixelKey == 1){
      if (annotation == 1){
        vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
        vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
        out_FragColor = vec4(color + sColor + aColor, 1.0);
      }
      else{
        vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
        out_FragColor = vec4(color + sColor, 1.0);
      }
    }
    else {
      if (prediction == 0 && annotation == 1){
            vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
            out_FragColor = vec4(color + aColor, 1.0);
      } else if (prediction == 1 && annotation == 1) {
            vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
            vec3 pColor = sampleTexture(predictionTexture, vPosition.xy).rgb;
            out_FragColor = vec4(color + aColor + pColor, 1.0);
      } else if (prediction == 0 && annotation == 1 && superpixel == 1) {
            vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
            vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
            out_FragColor = vec4(color + aColor + sColor, 1.0);
      } else if (prediction == 0 && annotation == 1 && superpixel == 1) {
            vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
            vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
            out_FragColor = vec4(color + aColor + sColor, 1.0);
      } else {
            out_FragColor = vec4(color, 1.0);
      }
    }

    if (confidence == 1 && annotation == 0 && prediction == 0 && superpixel == 0){
      vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
      out_FragColor = vec4(color + cColor, 1.0);
    }

    if (confidence == 1){
      if (annotation == 1){
        if (prediction == 1 && superpixel ==1){
          vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
          vec3 pColor = sampleTexture(predictionTexture, vPosition.xy).rgb;
          vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
          vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
          out_FragColor = vec4(color + aColor + sColor + pColor + cColor, 1.0);
        }
        else if (prediction == 1 && superpixel == 0){
          vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
          vec3 pColor = sampleTexture(predictionTexture, vPosition.xy).rgb;
          vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
          out_FragColor = vec4(color + aColor + pColor + cColor, 1.0);
        }
        else if (prediction == 0 && superpixel == 1){
          vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
          vec3 sColor = sampleTexture(superpixelTexture, vPosition.xy).rgb;
          vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
          out_FragColor = vec4(color + aColor + sColor + cColor, 1.0);
        }
        else{
          vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
          vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
          out_FragColor = vec4(color + cColor + aColor, 1.0);
        }
        
      }
      else {
        vec3 cColor = sampleTexture(confidenceTexture, vPosition.xy).rgb;
        out_FragColor = vec4(color + cColor, 1.0);
      }
    }

    // if (annotation == 1) {
    //     vec3 aColor = sampleTexture(annotationTexture, vPosition.xy).rgb;
    //     if (prediction == 1){
    //       vec3 pColor = sampleTexture(predictionTexture, vPosition.xy).rgb;
    //       out_FragColor = vec4(color + aColor + pColor, 1.0);
    //     }
    //     else{
    //       out_FragColor = vec4(color + aColor, 1.0);
    //     }
       
    // } else {
    //     out_FragColor = vec4(color, 1.0);
    // }
}
`
export const terrainShader = { _VS, _FS }
