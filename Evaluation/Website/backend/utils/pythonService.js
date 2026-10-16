// services/pythonService.js
const { spawn } = require("child_process");
const { execFile } = require("child_process");
const path = require("path");
var os = require("os");

//Create function to get CPU information
function cpuAverage() {

  //Initialise sum of idle and time of cores and fetch CPU info
  var totalIdle = 0, totalTick = 0;
  var cpus = os.cpus();

  //Loop through CPU cores
  for(var i = 0, len = cpus.length; i < len; i++) {

    //Select CPU core
    var cpu = cpus[i];

    //Total up the time in the cores tick
    for(type in cpu.times) {
      totalTick += cpu.times[type];
   }     

    //Total up the idle time of the core
    totalIdle += cpu.times.idle;
  }

  //Return the average Idle and Tick times
  return {idle: totalIdle / cpus.length,  total: totalTick / cpus.length};
}

//Grab first CPU Measure
var startMeasure = cpuAverage();

//Set delay for second Measure
setTimeout(function() { 

  //Grab second Measure
  var endMeasure = cpuAverage(); 

  //Calculate the difference in idle and total time between the measures
  var idleDifference = endMeasure.idle - startMeasure.idle;
  var totalDifference = endMeasure.total - startMeasure.total;

  //Calculate the average percentage CPU usage
  var percentageCPU = 100 - ~~(100 * idleDifference / totalDifference);

  //Output result to console
  console.log(percentageCPU + "% CPU Usage.");

}, 100);
exports.runCowPythonScript = (cowId) => {
    const scriptPath = path.join(__dirname, "..", "python", "generateGraph.py");
    const jsonPath   = path.join(__dirname, "..", "dataset.json");
    const graphPath  = path.join(__dirname, "..", "frontend", "public", "images", `graph${cowId}.png`);

    const pythonExec = path.join(__dirname, "..", "venv", "Scripts", "pythonw.exe");

    console.log("⚡ Ejecutando script Python en background…"); 

    try {
        const child = spawn(
            pythonExec,
            [scriptPath, jsonPath, cowId, graphPath],
            {
                windowsHide: true,
                shell: true,
                detached: true,
                stdio: "ignore"
            }
        );

        child.unref(); // Release Python so it runs in background

    } catch (err) {
        console.error("❌ ERROR ejecutando pythonw:", err);
    }
};


function runProgram(datasetPath, cowId, outputPath) {
    return new Promise((resolve, reject) => {
        const exePath = path.join(__dirname, "..", "..", "generateGraph.exe");

        execFile(
            exePath,
            [datasetPath, cowId, outputPath],  // <-- pass the 3 arguments
            { cwd: path.dirname(exePath) },
            (error, stdout, stderr) => {
                if (error) {
                    console.error("EXE error:", stderr || error);
                    return reject(error);
                }
                resolve(stdout);
            }
        );
        console.log("✔ EXE process started");
    });
}


module.exports = {
    runProgram
};