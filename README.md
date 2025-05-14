# Faster Annotation for Elevation-Guided Flood Extent Mapping by Consistency-Enhanced Active Learning      

# Instructions for installing heightmap meshing utility
Dependencies: 
1. C++11 or higher
2. glm: https://glm.g-truc.net/0.9.9/index.html

Installation:
1. on macOS: brew install glm # on macOS
   on Ubuntu / Debian: sudo apt-get install libglm-dev
2. git clone https://github.com/fogleman/hmm.git
3. cd hmm
4. make
5. make install

# Instructions for installing nodejs dependencies and packages
1. Install nodejs using this command: curl-o-https://raw.githubusercontent.com/nvm- sh/nvm/v0.39.5/install.sh | bash
2. Clone threejs repository: git clone https://github.com/Sean-Bradley/Three.js-TypeScript-Boilerplate.git
3. CD into folder: cd Three.js-TypeScript-Boilerplate
4. Install TypeScript: npm install -g typescript
5. Install dependencies: npm install
6. Start it: npm run dev
7. Visit http://127.0.0.1:8080, You should see a rotating green wireframe cube, and be able to rotate it further with your mouse.

# Instructions for running frontend code
1. git clone the repo
2. Copy folder named "node_modules" from Three.js-TypeScript-Boilerplate folder in the previous step and paste it into the code folder
3. cd into src/client folder: cd src/client
4. Start the client: npm run dev
5. Do not exit this terminal

# Instructions for running backend code
1. Open a new terminal
2. Install python environment using environment.yml file and activate the environment
3. cd into /alfa/backend_code/data_al folder
4. Run: python data_maker_al.py
5. cd into backend_code folder: cd ..
6. Run: flask run --port=5000
7. Do not exit this terminal

# Running the application in the browser
1. Open an up-to-date browser (Google Chrome prefered)
2. Navigate to chrome://settings/system and enable “Use hardware acceleration when available”, without this the application does not work!
3. Open this URL: localhost:8080
4. Enter your name on Student Id box and test_region_id on Test Region ID box.
5. Upload elevation data and rgb data and hit submit. You can find these files on ./alfa/backend_code/data_al/files_to_upload
6. Perform the experiments once you get the response from backend
7. The corresponding results with metrics will be stored inside /alfa/backend_code/user/<user_id>/output/Region_0_Metrics_C*.txt/ where the accuracy metrics for each cycle of Active Learning are stored. The metrics stored till line 12 on these files are our evaluation metrics.

# Demo Video  
https://youtu.be/jKJYLDIADaw 
